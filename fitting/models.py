import math
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import (
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
    CiqVariationalStrategy,
)

import gpytorch
import torch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from rich import print
import logging

logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float64)

SK = gpytorch.kernels.ScaleKernel


class RotParamMixin:
    def __init__(self, *args, rot_prior=None, rot_constraint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter(
            name="raw_rot",
            parameter=torch.nn.Parameter(torch.tensor(0.0, requires_grad=True)),
        )

        if rot_constraint is None:
            rot_constraint = gpytorch.constraints.Interval(-3.14, 3.14)

        self.register_constraint("raw_rot", rot_constraint)

        if rot_prior is not None:
            self.register_prior(
                "rot_prior", rot_prior, lambda m: m.rot, lambda m, v: m._rot_setter(v)
            )

    @property
    def rot(self):
        return self.raw_rot_constraint.transform(self.raw_rot)

    @rot.setter
    def rot(self, rot):
        self._rot_setter(rot)

    def _rot_setter(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rot)
        self.initialize(raw_rot=self.raw_rot_constraint.inverse_transform(value))

    def getMatrix(self):
        c = torch.cos(self.rot)
        s = torch.sin(self.rot)
        orth_mat = torch.stack((torch.stack([c, -s]), torch.stack([s, c])))
        return orth_mat


class RotMixin(RotParamMixin):
    def getDist(self, x1, x2, diag=False):
        diff = torch.unsqueeze(x1, dim=1) - x2
        m = self.getMatrix()
        d = torch.diag(1 / torch.squeeze(self.lengthscale) ** 2)
        real_mat = m.t() @ d @ m
        c = torch.einsum("abi,ij,abj->ab", diff, real_mat, diff)
        if diag:
            return c.diagonal()
        else:
            return c

    def forward(self, x1, x2, diag=False, **params):
        c = self.getDist(x1, x2, diag=diag)
        covar = self.post_function(c)
        return covar


class GeneralRQ(RotMixin, gpytorch.kernels.RQKernel):
    def post_function(self, dist_mat):
        alpha = self.alpha
        for _ in range(1, len(dist_mat.shape) - len(self.batch_shape)):
            alpha = alpha.unsqueeze(-1)
        return (1 + dist_mat.div(2 * alpha)).pow(-alpha)


class WrapLinear(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, idim=2, odim=2):
        super().__init__()
        self.transform = torch.nn.Linear(idim, odim)
        self.base_kernel = base_kernel

    def forward(self, x1, x2, **kwargs):
        return self.base_kernel(self.transform(x1), self.transform(x2), **kwargs)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, idim=2, odim=2, layer_sizes=(1000, 1000, 100)):
        super().__init__()
        for i in range(len(layer_sizes)):
            p = layer_sizes[i - 1] if i > 0 else idim
            self.add_module(f"linear{i}", torch.nn.Linear(p, layer_sizes[i]))
            self.add_module(f"relu{i}", torch.nn.ReLU())
            # if i == 1:
            #     self.add_module(f"dropout{i}", torch.nn.Dropout(p=0.2))
        self.add_module(
            f"linear{len(layer_sizes)}", torch.nn.Linear(layer_sizes[-1], odim)
        )


def wrapNN(cls_name, kernel):
    def __init__(
        self, *args, odim=None, idim=None, layer_sizes=None, nn=None, **kwargs
    ):
        kernel.__init__(self, *args, ard_num_dims=odim, **kwargs)
        if nn:
            self.feature_extractor = nn
        else:
            nnargs = dict(
                x
                for x in (("odim", odim), ("idim", idim), ("layer_sizes", layer_sizes))
                if x[1] is not None
            )
            self.feature_extractor = LargeFeatureExtractor(**nnargs)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, x1, x2, **params):
        # x1_, x2_ = (
        #    self.scale_to_bounds(x1),
        #    self.scale_to_bounds(x2),
        # )
        x1_, x2_ = (
            self.feature_extractor(x1),
            self.feature_extractor(x2),
        )
        x1_, x2_ = (
            self.scale_to_bounds(x1_),
            self.scale_to_bounds(x2_),
        )
        return kernel.forward(self, x1_, x2_, **params)

    return type(cls_name, (kernel,), dict(__init__=__init__, forward=forward))


NNSMKernel = wrapNN("NNSMKernel", gpytorch.kernels.SpectralMixtureKernel)
NNRBFKernel = wrapNN("NNRBFKernel", gpytorch.kernels.RBFKernel)
NNRQKernel = wrapNN("NNRQKernel", gpytorch.kernels.RQKernel)
NNMaternKernel = wrapNN("NNMaternKernel", gpytorch.kernels.MaternKernel)


class RBFLayer(torch.nn.Module):
    def __init__(self, dim, count):
        super().__init__()
        self.log_length_scales = torch.nn.Parameter(torch.rand(count, dim) + 0.01)
        self.centers = torch.nn.Parameter(torch.rand(count, dim))

    def forward(self, vals):
        # print(torch.unsqueeze(vals, 1).shape)
        return torch.exp(
            -(
                (torch.unsqueeze(vals, 1) - self.centers)
                / (2 * torch.exp(self.log_length_scales))
            )
            .pow(2)
            .sum(-1)
        )


class NonStatKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, base_kernel=None, dim=2, count=4, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.pre_transform = RBFLayer(dim, count)
        self.trans = torch.nn.Linear(count, 1, bias=bias)
        if base_kernel is None:
            # base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=dim)
            # base_kernel = GeneralRBF(ard_num_dims=dim)
            # base_kernel = NNRBFKernel(odim=1,idim=dim, layer_sizes=(4,4))
            base_kernel = NNRBFKernel(odim=1, idim=dim, layer_sizes=(2,))
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        # for m,p in self.named_parameters():
        #     print(f"{m} = {p}")

        # r = super().forward(x1, x2, diag=diag, **params)
        r = self.base_kernel.forward(x1, x2, diag=diag, **params)
        v1 = self.trans(self.pre_transform(x1))
        v2 = self.trans(self.pre_transform(x2))
        # print(v1)
        # print(v2)
        if diag:
            o = torch.squeeze(v1 * v2)
        else:
            o = torch.outer(v1[..., 0], v2[..., 0])

        return o * r


class CenteredKernelWrapper(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, center=None, dim=2, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        if center is not None:
            self.center = torch.nn.Parameter(center)
        else:
            self.center = torch.nn.Parameter(torch.zeros((dim,)))

    def forward(self, x1, x2, **kwargs):
        return self.base_kernel(x1 - self.center, x2 - self.center, **kwargs)


class NonStatParametric2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_ratio=2):
        super().__init__(train_x, train_y, likelihood)
        self.inducing_ratio = inducing_ratio
        self.mean_module = gpytorch.means.ConstantMean()
        # c = NonStatKernel(ard_num_dims=2, count=1)

        # self.base_covar_module = c  # + gpytorch.kernels.RBFKernel(ard_num_dims=2)

        self.base_covar_module = CenteredKernelWrapper(
            gpytorch.kernels.RBFKernel(ard_num_dims=2),
            center=torch.Tensor([[0.3, 0.3]]),
        )

        self.base_covar_module.center.requires_grad = False

        # self.covar_module = gpytorch.kernels.InducingPointKernel(
        #     self.base_covar_module, likelihood=likelihood, inducing_points=ind
        # )
        self.covar_module = self.base_covar_module

        # self.covar_module.inducing_points.requires_grad_(False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyNNRBFModel2D(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, inducing_ratio=3, num_inducing=None
    ):

        if num_inducing and inducing_ratio:
            raise RuntimeError("Cannot have both num inducing and inducing ratio")

        super().__init__(train_x, train_y, likelihood)
        if inducing_ratio:
            self.inducing_ratio = inducing_ratio
            ind = train_x[:: self.inducing_ratio].clone()
            self.num_inducing = ind.size(0)
        else:
            self.inducing_ratio = None
            self.num_inducing = num_inducing
            ind = train_x[:num_inducing].clone()

        self.mean_module = gpytorch.means.ConstantMean()

        base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(12, 12, 8)))
        # + SK(
        #     gpytorch.kernels.MaternKernel(
        #         mu=2.5,
        #         ard_num_dims=2,
        #         lengthscale_constraint=gpytorch.constraints.Interval(0.5, 20.0),
        #     )
        # )
        # + SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(12, 8, 4)))
        # + SK(
        #     gpytorch.kernels.MaternKernel(
        #         ard_num_dims=2,
        #         mu=2.5,
        #         # lengthscale_constraint=gpytorch.constraints.Interval(0.0, 0.2),
        #     )
        # )
        # base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(12, 8)))
        # base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(12, 12, 8)))
        # base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(50, 50, 25)))
        # base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(100, 100, 50)))
        # + SK(
        #     gpytorch.kernels.MaternKernel(
        #         ard_num_dims=2,
        #         mu=1.5,
        #         lengthscale_constraint=gpytorch.constraints.Interval(0.0, 0.2),
        #     )
        # )
        # base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(12, 8, 4)))
        # base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(20, 20, 10)))
        # base_covar_module = SK(NNRBFKernel(idim=2, odim=2, layer_sizes=(100, 50, 25)))
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_covar_module, likelihood=likelihood, inducing_points=ind
        )
        # self.covar_module = base_covar_module

    def forward(self, x):
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyRBFModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, inducing_ratio=1, num_inducing=None
    ):

        if num_inducing and inducing_ratio:
            raise RuntimeError("Cannot have both num inducing and inducing ratio")

        super().__init__(train_x, train_y, likelihood)
        if inducing_ratio:
            self.inducing_ratio = inducing_ratio
            ind = train_x[:: self.inducing_ratio].clone()
            self.num_inducing = ind.size(0)
        else:
            self.inducing_ratio = None
            self.num_inducing = num_inducing
            ind = train_x[:num_inducing].clone()

        self.mean_module = gpytorch.means.ZeroMean()

        base_covar_module = SK(gpytorch.kernels.RBFKernel(ard_num_dims=2))

        self.covar_module = base_covar_module

    def forward(self, x):
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
