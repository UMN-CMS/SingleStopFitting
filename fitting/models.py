import math

import gpytorch
import torch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from rich import print


class GaussianMean(gpytorch.means.Mean):
    def __init__(self, prior=None, init_mean=0.0, init_sigma=1.0, init_scale=1.0):
        super().__init__()
        self.register_parameter(
            name="mean",
            parameter=torch.nn.Parameter(torch.tensor(init_mean, dtype=torch.float64)),
        )
        self.register_parameter(
            name="sigma",
            parameter=torch.nn.Parameter(torch.tensor(init_sigma, dtype=torch.float64)),
        )
        self.register_parameter(
            name="scale",
            parameter=torch.nn.Parameter(torch.tensor(init_scale, dtype=torch.float64)),
        )
        if prior is not None:
            self.register_prior("contant_prior", prior, "mean")
            self.register_prior("constant_prior", prior, "sigma")
            self.register_prior("constant_prior", prior, "scale")

    def forward(self, input):
        inner = (input - self.mean) ** 2 @ (1 / self.sigma**2)
        e = torch.exp(-inner)
        ret = self.scale * e
        return ret


class KnownMean(gpytorch.means.Mean):
    def __init__(self, vals):
        super().__init__()
        self.vals = torch.nn.Parameter(vals, requires_grad=False)
        self.register_parameter(name="mean", parameter=self.vals)

    def forward(self, x):
        print(self.vals.shape)
        print(x.shape)
        # r= self.vals.expand(torch.broadcast_shapes(self.vals.shape, x.shape[:-1]))
        # print(r.shape)
        return self.vals


class GPMean(gpytorch.means.Mean):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        self.gp.eval()
        return self.gp(x)


class HeterogenousConstantMean(gpytorch.means.Mean):
    def __init__(self, init, prior=None):
        super().__init__()
        self.register_parameter(name="values", parameter=torch.nn.Parameter(init))
        if prior is not None:
            self.register_prior("contant_prior", prior, "mean")

    def forward(self, input):
        return self.values


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

    def __init__(self, train_x, train_y, likelihood, *args, **kwargs):
        super(RotMixin,self).__init__(train_x, train_y, *args,**kwargs)
        


class FunctionRBF(GeneralRBF):
    is_stationary = False

    def __init__(self, function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = function

    def forward(self, x1, x2, **params):
        ret = torch.einsum(
            "ik,jk->ij",
            self.function(x1),
            self.function(x2),
        )
        print(ret.size())
        f = super().forward(x1, x2)
        print(f.size())
        ret = ret * f
        print(f)
        return ret


class GeneralMatern(RotMixin, gpytorch.kernels.MaternKernel):
    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.getDist(x1_, x2_, diag=diag)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (
                    (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
                )
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.getProduct(x1, x2)
        )


class GeneralSpectralMixture(RotParamMixin, gpytorch.kernels.SpectralMixtureKernel):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(
                    self.ard_num_dims, num_dims
                )
            )

        # Expand x1 and x2 to account for the number of mixtures
        # Should make x1/x2 (... x k x n x d) for k mixtures
        print(f"X1 shape is {x1.shape}")
        x1_ = x1.unsqueeze(-3)
        x2_ = x2.unsqueeze(-3)
        print(f"X1_ shape is {x1_.shape}")

        # Compute distances - scaled by appropriate parameters
        x1_exp = x1_ * self.mixture_scales
        x2_exp = x2_ * self.mixture_scales
        x1_cos = x1_ * self.mixture_means
        x2_cos = x2_ * self.mixture_means

        # Create grids
        x1_exp_, x2_exp_ = self._create_input_grid(x1_exp, x2_exp, diag=diag, **params)
        x1_cos_, x2_cos_ = self._create_input_grid(x1_cos, x2_cos, diag=diag, **params)

        exp_diff = x1_exp_ - x2_exp_
        cos_diff = x1_cos_ - x2_cos_
        print(diag)
        print(exp_diff.shape)

        m = self.getMatrix()
        real_mat = self.getMatrix()
        exp_val = torch.einsum("cabi,ij,cabj->cab", exp_diff, real_mat, exp_diff)

        # print(self.mixture_means.shape)
        # Compute the exponential and cosine terms
        exp_term = exp_val.mul_(-2 * math.pi**2)
        # exp_term = exp_diff.pow_(2).mul_(-2 * math.pi**2)
        cos_term = cos_diff.mul_(2 * math.pi)
        exp_term = torch.unsqueeze(exp_term, 3)
        cos_term = torch.unsqueeze(cos_term.sum(3), 3)
        # print(exp_term.shape)
        # print(cos_term.shape)
        res = exp_term.exp_() * cos_term.cos_()

        # Sum over mixtures
        mixture_weights = self.mixture_weights.view(*self.mixture_weights.shape, 1, 1)
        if not diag:
            mixture_weights = mixture_weights.unsqueeze(-2)

        res = (res * mixture_weights).sum(-3 if diag else -4)

        # Product over dimensions
        if last_dim_is_batch:
            # Put feature-dimension in front of data1/data2 dimensions
            res = res.permute(*list(range(0, res.dim() - 3)), -1, -3, -2)
        else:
            res = res.prod(-1)

        return res


class ExactProjGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.register_parameter
        # self.proj_mat = torch.nn.Parameter(torch.tensor([[1,1],[1,0]], dtype=torch.float64))
        self.rot = torch.nn.Parameter(torch.tensor(0.78, dtype=torch.float64))

    def forward(self, x):
        rot_mat = torch.tensor(
            [
                [torch.cos(self.rot), -torch.sin(self.rot)],
                [torch.sin(self.rot), torch.cos(self.rot)],
            ]
        )
        x = x @ rot_mat  # n x d * d x k --> n x k
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = mean or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactAnyKernelModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None, mean=None):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        if mean is None:
            mean = gpytorch.means.ZeroMean()
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalAnyKernelModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel=None, mean=None):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = mean or gpytorch.means.ConstantMean()
        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactPeakedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

        self.covar_peak_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.peak = torch.nn.Parameter(torch.tensor([0.8, 0.2], dtype=torch.float64))

    def forward(self, x):
        subbed = x - self.peak
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) * self.covar_peak_module(subbed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PeakedMixin:
    is_stationary = False

    def __init__(self, *args, peak_prior=None, peak_constraint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter(
            name="raw_peak",
            parameter=torch.nn.Parameter(torch.tensor([0.1, 0.1], requires_grad=True)),
        )
        if peak_constraint is None:
            peak_constraint = gpytorch.constraints.Interval(0.0, 1.0)
        self.register_constraint("raw_peak", peak_constraint)
        if peak_prior is not None:
            self.register_prior(
                "peak_prior",
                peak_prior,
                lambda m: m.peak,
                lambda m, v: m._set_peak(v),
            )

    @property
    def peak(self):
        return self.raw_peak_constraint.transform(self.raw_peak)

    @peak.setter
    def peak(self, value):
        return self._set_peak(value)

    def _set_peak(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_peak)
        self.initialize(raw_peak=self.raw_peak_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
        return super().forward(x1 - self.peak, x2 - self.peak, **params)


class PeakedGRBF(PeakedMixin, GeneralRBF):
    pass


class PeakedRBF(PeakedMixin, gpytorch.kernels.RBFKernel):
    pass


class PeakedSMK(PeakedMixin, gpytorch.kernels.SpectralMixtureKernel):
    pass


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


class InducingPointModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, kernel=None, inducing=None, mean=None
    ):
        super().__init__(train_x, train_y, likelihood)
        if mean is None:
            mean = gpytorch.means.ZeroMean()
        self.mean_module = mean
        self.base_covar_module = kernel
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing.clone(),
            likelihood=likelihood,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class KISSModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x, 1.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                kernel, grid_size=grid_size, num_dims=1
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PyroGPModel(gpytorch.models.PyroGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        kernel=None,
        mean=None,
        num_inducing=None,
        inducing_points=None,
    ):
        # variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
        #     inducing_points.size(0)
        # )
        # variational_strategy = gpytorch.variational.CiqVariationalStrategy(
        #     self,
        #     inducing_points,
        #     variational_distribution,
        #     learn_inducing_locations=True,
        # )
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(
            variational_strategy,
            likelihood,
            num_data=train_y.numel(),
            name_prefix="simple_regression_model",
        )
        self.likelihood = likelihood
        self.mean_module = mean or gpytorch.means.ConstantMean()
        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )
        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)  # Returns an n_data vec
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


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
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=dim)
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        # for m,p in self.named_parameters():
        #     print(f"{m} = {p}")

        # r = super().forward(x1, x2, diag=diag, **params)
        r = self.base_kernel.forward(x1, x2, diag=diag, **params)
        v1 = self.trans(self.pre_transform(x1))
        v2 = self.trans(self.pre_transform(x2))
        if diag:
            o = torch.squeeze(v1 * v2)
        else:
            o = torch.outer(v1.squeeze(), v2.squeeze())

        return o * r


class NonStatParametric2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, function=None, num_inducing=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = (
            NonStatKernel(ard_num_dims=2)
            + NonStatKernel(ard_num_dims=2)
            + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
        )
        if num_inducing is None:
            ind = train_x[::4].clone()
        else:
            ind = train_x[:num_inducing].clone()
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module, likelihood=likelihood, inducing_points=ind
        )

        # self.covar_module.inducing_points.requires_grad_(False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NonStatParametric1D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, function=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = (
            # NonStatKernel(count=2, ard_num_dims=1, dim=1, bias=False)
            NonStatKernel(count=1, ard_num_dims=1, dim=1)
            +
            # NonStatKernel(base_kernel=gpytorch.kernels.RBFKernel(), count=1, dim=1)
            # NonStatKernel(count=2, ard_num_dims=1, dim=1)
            # gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=1))
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))
            # + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))
            # gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=1))
            # + gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=1))
            # gpytorch.kernels.SpectralMixtureKernel(num_mixtures=8, ard_num_dims=1)
        )
        # self.base_covar_module.kernels[0].pre_transform.centers.data= torch.tensor([0.1])
        # self.base_covar_module.kernels[0].pre_transform.centers.data._requires_grad(False)

        self.covar_module = self.base_covar_module
        # self.covar_module = gpytorch.kernels.InducingPointKernel(
        #     self.base_covar_module,
        #     likelihood=likelihood,
        #     inducing_points=train_x[::].clone(),
        # )

        # self.covar_module.inducing_points.requires_grad_(True)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyNNRBFModel2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_inducing=None):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = LargeFeatureExtractor(
            odim=2, idim=2, layer_sizes=(100,50)
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        if num_inducing is None:
            ind = train_x[::2].clone()
        else:
            ind = train_x[:num_inducing].clone()

        print(ind.shape)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            likelihood=likelihood,
            inducing_points=ind,
        )

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


class MyRBF2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            likelihood=likelihood,
            inducing_points=train_x[::4].clone(),
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
