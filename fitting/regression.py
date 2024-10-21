import contextlib
import copy
from collections import namedtuple
from dataclasses import dataclass

import gpytorch
import torch
from rich import print
from rich.progress import Progress

from .models import ExactAnyKernelModel
from .utils import chi2Bins

DataValues = namedtuple("DataValues", "X Y V E")


@dataclass
class DataValues:
    X: torch.Tensor
    Y: torch.Tensor
    V: torch.Tensor
    E: torch.Tensor

    def toGpu(self):
        return DataValues(self.X.cuda(), self.Y.cuda(), self.V.cuda(), self.E)

    def fromGpu(self):
        return DataValues(self.X.cpu(), self.Y.cpu(), self.V.cpu(), self.E)


def makeRegressionData(
    histogram,
    mask_function=None,
    exclude_less=None,
    get_mask=False,
    get_shaped_mask=False,
    domain_mask_function=None,
    get_window_mask=False,
    #get_reshape_function=False,
    get_derivatives=False,
):
    if mask_function is None:
        mask_function = lambda x1, x2: (torch.full_like(x1, False, dtype=torch.bool))

    edges_x1 = torch.from_numpy(histogram.axes[0].edges)
    edges_x2 = torch.from_numpy(histogram.axes[1].edges)

    centers_x1 = torch.diff(edges_x1) / 2 + edges_x1[:-1]
    centers_x2 = torch.diff(edges_x2) / 2 + edges_x2[:-1]

    bin_values = torch.from_numpy(histogram.values()).T
    bin_vars = torch.from_numpy(histogram.variances()).T
    centers_grid_x1, centers_grid_x2 = torch.meshgrid(
        centers_x1, centers_x2, indexing="xy"
    )
    if exclude_less:
        domain_mask = bin_values < exclude_less
    else:
        domain_mask = torch.full_like(bin_values, False, dtype=torch.bool)

    centers_grid = torch.stack((centers_grid_x1, centers_grid_x2), axis=2)

    if domain_mask_function is not None:
        domain_mask = domain_mask | domain_mask_function(
            centers_grid[:, :, 0], centers_grid[:, :, 1]
        )

    m = mask_function(centers_grid[:, :, 0], centers_grid[:, :, 1])
    centers_mask = m | domain_mask
    flat_centers = torch.flatten(centers_grid, end_dim=1)
    flat_bin_values = torch.flatten(bin_values)
    flat_bin_vars = torch.flatten(bin_vars)
    ret = DataValues(
        flat_centers[torch.flatten(~centers_mask)],
        flat_bin_values[torch.flatten(~centers_mask)],
        flat_bin_vars[torch.flatten(~centers_mask)],
        (edges_x1, edges_x2),
    )
    ret = (ret,)


    if get_mask:
        ret = (*ret, torch.flatten(~centers_mask))
    if get_shaped_mask:
        ret = (*ret, centers_mask)

    if get_derivatives:
        dx,dy =torch.gradient(bin_values,spacing=(centers_x1,centers_x2))
        flat_dx = torch.flatten(dx)[torch.flatten(~centers_mask)]
        flat_dy = torch.flatten(dy)[torch.flatten(~centers_mask)]
        ret = (*ret,(flat_dx,flat_dy))
    return ret


def createModel(train_data, kernel=None, model_maker=None, learn_noise=False, **kwargs):
    v = train_data.V

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=v,
        learn_additional_noise=learn_noise,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if model_maker is None:
        model_maker = ExactAnyKernelModel

    if kernel:
        model = model_maker(
            train_data.X, train_data.Y, likelihood, kernel=kernel, **kwargs
        )
    else:
        model = model_maker(train_data.X, train_data.Y, likelihood, **kwargs)
    return model, likelihood


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def optimizeHyperparams(
    model,
    likelihood,
    train_data,
    iterations=100,
    bar=True,
    lr=0.05,
    get_evidence=False,
    mll=None,
    chi2mask=None,
    val=None,
):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    if mll is None:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=iterations // 3, gamma=0.1
    # )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, 0.1, 0.001, step_size_up=100,
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100,)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    context = Progress() if bar else contextlib.nullcontext()
    evidence = None

    k = torch.kthvalue(train_data.Y, int(train_data.Y.size(0) * 0.1)).values
    m = train_data.Y > k

    slr = lr
    with context as progress:
        if bar:
            task1 = progress.add_task("[red]Optimizing...", total=iterations)
        # def closure():
        #     optimizer.zero_grad()
        #     output = model(train_data.X)
        #     ml = -mll(output, train_data.Y)
        #     ml.backward()
        #     return ml
        for i in range(iterations):
            optimizer.zero_grad()

            output = model(train_data.X)
            ml = -mll(output, train_data.Y)
            loss = ml
            # optimizer.step(closure)

            # scheduler.step()
            # scheduler.step(loss)
            # slr = get_lr(optimizer)
            # slr = scheduler.get_last_lr()[0]

            # if slr < 1e-3 * lr:
            #    print(
            #        f"Iter {i} (lr={slr}): Loss = {round(loss.item(),4)}"
            #    )
            #    evidence = float(loss.item())
            #    break

            if bar:
                progress.update(
                    task1,
                    advance=1,
                    description=f"[red]Optimizing(Loss is {round(loss.item(),3)})...",
                )
                progress.refresh()
            else:
                if (i % (iterations // 20) == 0) or i == iterations - 1:
                    model.eval()
                    if val is not None:
                        v = val(model)
                    output = model(train_data.X)
                    model.train()
                    chi2 = chi2Bins(
                        output.mean, train_data.Y, train_data.V, mask=chi2mask
                    ).item()
                    chi2_p = chi2Bins(output.mean, train_data.Y, output.variance).item()
                    s = (
                        f"Iter {i} (lr={slr:0.4f}): MLL={round(loss.item(),4)},"
                        f"X2/B={chi2:0.2f}, "
                        f"X2P/B={chi2_p:0.2f}"
                    )
                    if val is not None:
                        s += f" Val={v:0.2f}"
                    for n, p in model.named_parameters():
                        x = p.round(decimals=2).tolist()
                        if not isinstance(x, list) or len(x) < 4:
                            print(f"{n} = {x}")
                    ls = None
                    if hasattr(model.covar_module.base_kernel, "lengthscale"):
                        ls = model.covar_module.base_kernel.lengthscale
                    elif hasattr(
                        model.covar_module.base_kernel.base_kernel, "lengthscale"
                    ):
                        ls = model.covar_module.base_kernel.base_kernel.lengthscale

                    if ls is not None:
                        print(f"lengthscale = {ls.round(decimals=2).tolist()}")

                    print(s)

                    evidence = float(loss.item())
                    pass
            loss.backward()
            optimizer.step()

    if get_evidence:
        return model, likelihood, evidence
    else:
        return model, likelihood


def getPrediction(model, likelihood, test_data):
    with torch.no_grad():
        observed_pred = model(test_data.X)
    return observed_pred


def getBlindedMask(inputs, pred_mean, test_mean, test_var, mask_func):
    mask = mask_func(inputs[:, 0], inputs[:, 1])
    return mask
    # pred_mean = pred_mean[mask]
    # test_mean = test_mean[mask]
    # test_var = test_var[mask]
    # num = torch.count_nonzero(mask)
    # return torch.sum((test_mean - pred_mean) ** 2 / test_var) / num
