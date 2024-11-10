import contextlib
from collections import namedtuple
from dataclasses import dataclass

import gpytorch
import torch
from rich import print
from rich.progress import Progress

from .models import ExactAnyKernelModel
from .utils import chi2Bins, dataToHist

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

    @property
    def dim(self):
        return len(self.E)

    def toHist(self):
        return dataToHist(self.X,self.Y,self.E,self.V)


def makeRegressionData(
    histogram,
    mask_function=None,
    exclude_less=None,
    get_mask=False,
    get_shaped_mask=False,
    domain_mask_function=None,
    get_window_mask=False,
):
    if mask_function is None:
        mask_function = lambda *x: (torch.full_like(x[0], False, dtype=torch.bool))

    edges = tuple(torch.from_numpy(a.edges) for a in histogram.axes)
    centers = tuple(torch.diff(e) / 2 + e[:-1] for e in edges)
    bin_values = torch.from_numpy(histogram.values())
    bin_vars = torch.from_numpy(histogram.variances())
    if len(edges) == 2:
        bin_values = bin_values.T
        bin_vars = bin_vars.T
        
    centers_grid = torch.meshgrid(*centers, indexing="xy")
    if exclude_less:
        domain_mask = bin_values < exclude_less
    else:
        domain_mask = torch.full_like(bin_values, False, dtype=torch.bool)

    centers_grid = torch.stack(centers_grid, axis=-1)
    unbound = torch.unbind(centers_grid, dim=-1)
    if domain_mask_function is not None:
        domain_mask = domain_mask | domain_mask_function(*unbound)
    m = mask_function(*unbound)
    centers_mask = m | domain_mask
    flat_centers = torch.flatten(centers_grid, end_dim=1)
    flat_bin_values = torch.flatten(bin_values)
    flat_bin_vars = torch.flatten(bin_vars)
    ret = DataValues(
        flat_centers[torch.flatten(~centers_mask)],
        flat_bin_values[torch.flatten(~centers_mask)],
        flat_bin_vars[torch.flatten(~centers_mask)],
        edges,
    )
    ret = (ret,)
    if get_mask:
        ret = (*ret, torch.flatten(~centers_mask))
    if get_shaped_mask:
        ret = (*ret, centers_mask)
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
    if mll is None:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=iterations // 1, gamma=0.1
    )
    context = Progress() if bar else contextlib.nullcontext()
    evidence = None
    k = torch.kthvalue(train_data.Y, int(train_data.Y.size(0) * 0.05)).values
    m = train_data.Y > k
    slr = lr

    def closure():
        optimizer.zero_grad()
        output = model(train_data.X)
        loss = -mll(output, train_data.Y)
        loss = loss - loocv(output, train_data.Y)
        loss.backward()
        return loss

    for i in range(iterations):
        optimizer.zero_grad()
        output = model(train_data.X)
        loss = -mll(output, train_data.Y)
        # loss = -loocv(output, train_data.Y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        slr = scheduler.get_last_lr()[0]

        if (i % (iterations // 20) == 0) or i == iterations - 1:
            model.eval()
            if val is not None:
                v = val(model)
            output = model(train_data.X)
            model.train()
            chi2 = chi2Bins(
                output.mean, train_data.Y, train_data.V, mask=chi2mask
            )  # .item()

            # loss =  loss + abs(1 - chi2)
            chi2_p = chi2Bins(output.mean, train_data.Y, output.variance).item()
            s = (
                f"Iter {i} (lr={slr:0.4f}): Loss={round(loss.item(),4)},"
                f"X2/B={chi2.item():0.2f}, "
                f"X2P/B={chi2_p:0.2f}"
            )
            if val is not None:
                s += f" Val={v:0.2f}"
            for n, p in model.named_parameters():
                x = p.flatten().round(decimals=2).tolist()
                if not isinstance(x, list) or len(x) < 4:
                    print(f"{n} = {x}")
            ls = None
            try:
                if hasattr(model.covar_module.base_kernel, "lengthscale"):
                    ls = model.covar_module.base_kernel.lengthscale
                elif hasattr(model.covar_module.base_kernel.base_kernel, "lengthscale"):
                    ls = model.covar_module.base_kernel.base_kernel.lengthscale
            except Exception as e:
                pass

            if ls is not None:
                print(f"lengthscale = {ls.round(decimals=2).tolist()}")

            print(s)

            evidence = float(loss.item())
            # if chi2 < 1.05 and i > 20:
            #     break

    if get_evidence:
        return model, likelihood, evidence
    else:
        return model, likelihood


def getPrediction(model, likelihood, test_data):
    with torch.no_grad():
        observed_pred = model(test_data.X)
    return observed_pred


def getBlindedMask(inputs, mask_func):
    if inputs.dim() > 1:
        unbound = torch.unbind(inputs, dim=-1)
    else:
        unbound = (inputs,)
    mask = mask_func(*unbound)
    return mask
