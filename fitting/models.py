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


class GeneralRBF(RotMixin, gpytorch.kernels.RBFKernel):
    def post_function(self, dist_mat):
        return gpytorch.kernels.rbf_kernel.postprocess_rbf(dist_mat)

    # def __init__(self, train_x, train_y, likelihood, *args, **kwargs):
    #     super(RotMixin,self).__init__(train_x, train_y, *args,**kwargs)



class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, idim=2, odim=2, layer_sizes=(1000, 1000, 100)):
        super().__init__()
        for i in range(len(layer_sizes)):
            p = layer_sizes[i - 1] if i > 0 else idim
            self.add_module(f"linear{i}", torch.nn.Linear(p, layer_sizes[i]))
            self.add_module(f"relu{i}", torch.nn.ReLU())
        self.add_module(
            f"linear{len(layer_sizes)}", torch.nn.Linear(layer_sizes[-1], odim)
        )

def wrapNN(cls_name, kernel):
    def __init__(
        self, *args, odim=None, idim=None, layer_sizes=None, nn=None, **kwargs
    ):
        kernel.__init__(self, *args, **kwargs, ard_num_dims=odim)
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
NNGRBFKernel = wrapNN("NNGRBFKernel", GeneralRBF)
NNRQKernel = wrapNN("NNRQKernel", gpytorch.kernels.RQKernel)
NNMaternKernel = wrapNN("NNMaternKernel", gpytorch.kernels.MaternKernel)


class RBFLayer(torch.nn.Module):
    def __init__(self, dim, count):
        super().__init__()
        self.length_scales = torch.nn.Parameter(torch.rand(count, dim) + 0.01)
        self.centers = torch.nn.Parameter(torch.rand(count, dim))

    def forward(self, vals):
        # print(torch.unsqueeze(vals, 1).shape)
        return torch.exp(
            -((torch.unsqueeze(vals, 1) - self.centers) / (2 * self.length_scales))
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
            o = torch.outer(v1.squeeze(), v2.squeeze())

        return o * r



class NonStatParametric2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_ratio=2):
        super().__init__(train_x, train_y, likelihood)
        self.inducing_ratio = inducing_ratio
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = (
            NonStatKernel(ard_num_dims=2, count=5)
            + NonStatKernel(ard_num_dims=2, count=5)
            + gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        ind = train_x[:: self.inducing_ratio].clone()
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module, likelihood=likelihood, inducing_points=ind
        )

        # self.covar_module.inducing_points.requires_grad_(False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyNNRBFModel2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_ratio=4):
        super().__init__(train_x, train_y, likelihood)
        self.inducing_ratio = inducing_ratio
        # self.feature_extractor = LargeFeatureExtractor(
        #     odim=2, idim=2, layer_sizes=(40, 20)
        # )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        ind = train_x[:: self.inducing_ratio].clone()

        self.mean_module = gpytorch.means.ZeroMean()
        # self.base_covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=2)
        # )

        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            NNRBFKernel(idim=2, odim=2, layer_sizes=(16, 8))
        )

        # self.base_covar_module = gpytorch.kernels.ScaleKernel(
        #     NNMaternKernel(idim=2, odim=2, layer_sizes=(4,)), mu=2.5
        # )

        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            likelihood=likelihood,
            inducing_points=ind,
        )

    def forward(self, x):
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyVariational2DModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, likelihood, inducing_ratio=1):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))

        self.inducing_ratio = inducing_ratio

        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.likelihood = likelihood
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        self.mean_module = gpytorch.means.ZeroMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            # gpytorch.kernels.RBFKernel(ard_num_dims=2)
            NNMaternKernel(idim=2, odim=2, layer_sizes=(16, 8), mu=1.5)
        )

    def forward(self, x):
        # x = self.feature_extractor(x)
        # x = self.scale_to_bounds(x)
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyNNSpectralModel2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_inducing=None):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = LargeFeatureExtractor(
            odim=2, idim=2, layer_sizes=(40, 20)
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        if num_inducing is None:
            ind = train_x[::4].clone()
        else:
            ind = train_x[:num_inducing].clone()

        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.SpectralMixtureKernel(n_mixtures=4)
        self.base_covar_module.initialize_from_data(train_x, train_y)

        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            likelihood=likelihood,
            inducing_points=ind,
        )

        # def init_weights(m):
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)

        # self.covar_module.inducing_points.requires_grad_(False)

        # self.apply(init_weights)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.scale_to_bounds(x)
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyRBF2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_ratio=4):
        super().__init__(train_x, train_y, likelihood)
        self.inducing_ratio = inducing_ratio
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            likelihood=likelihood,
            inducing_points=train_x[:: self.inducing_ratio].clone(),
        )

    def forward(self, x):
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyNNRBFModel1D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = LargeFeatureExtractor(
            odim=1, idim=1, layer_sizes=(20, 20)
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.covar_module = self.base_covar_module
        # self.covar_module = gpytorch.kernels.InducingPointKernel(
        #     self.base_covar_module,
        #     likelihood=likelihood,
        #     inducing_points=train_x[::4].clone(),
        # )

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        # self.covar_module.inducing_points.requires_grad_(False)

        self.apply(init_weights)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.scale_to_bounds(x)
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
