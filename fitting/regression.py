import contextlib
import code
import logging
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import fitting.models
import fitting.transformations as transformations
import gpytorch
import hist
import numpy as np
import torch
import uproot
from fitting.utils import chi2Bins, getScaledEigenvecs, modelToPredMVN
from rich import print
from rich.progress import Progress

from .models import ExactAnyKernelModel
from .utils import chi2Bins, dataToHist
import json
import logging
import pickle as pkl
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import gpytorch
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import torch
from gpytorch.kernels import ScaleKernel as SK
from rich import print

from . import models, regression, transformations, windowing
from .plots import makeDiagnosticPlots, makeNNPlots
from .predictive import makePosteriorPred
from .utils import chi2Bins, modelToPredMVN, dataToHist


@dataclass
class TrainedModel:
    model_class: Any
    model_state: dict

    input_data: hist.Hist

    domain_mask: torch.Tensor
    blind_mask: torch.Tensor

    transform: torch.Tensor
    metadata: dict


def getPrediction(trained_model, other_data=None):
    model_class = trained_model.model_class
    model_state = trained_model.model_state

    hist = trained_model.input_data
    raw_regression_data = DataValues.fromHistogram(trained_model.input_data)

    print(raw_regression_data.X.shape)
    bm = ~trained_model.blind_mask
    print(torch.count_nonzero(bm))
    dm = trained_model.domain_mask

    if other_data is None:
        all_data = raw_regression_data.getMasked(dm)
    else:
        all_data = other_data
    if bm is None:
        bm = torch.zeros_like(all_data.Y, dtype=bool)

    blinded_data = all_data.getMasked(bm)

    print(blinded_data.X.shape)

    transform = trained_model.transform
    normalized_blinded_data = transform.transform(blinded_data)
    normalized_all_data = transform.transform(all_data)

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=normalized_blinded_data.V,
        learn_additional_noise=False,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )
    model = model_class(
        normalized_blinded_data.X, normalized_blinded_data.Y, likelihood
    )
    model.load_state_dict(model_state)

    model.eval()
    pred_dist = modelToPredMVN(
        model,
        model.likelihood,
        normalized_all_data,
        slope=transform.transform_y.slope,
        intercept=transform.transform_y.intercept,
    )
    pred_data = DataValues(all_data.X, pred_dist.mean, pred_dist.variance, all_data.E)
    return model, transform, all_data, pred_dist


@dataclass
class DataValues:
    X: torch.Tensor
    Y: torch.Tensor
    V: torch.Tensor
    E: torch.Tensor

    def getMasked(self, mask):
        return DataValues(self.X[mask], self.Y[mask], self.V[mask], self.E)

    def __getitem__(self, m):
        return self.getMasked(m)

    def toGpu(self):
        return DataValues(self.X.cuda(), self.Y.cuda(), self.V.cuda(), self.E)

    def fromGpu(self):
        return DataValues(self.X.cpu(), self.Y.cpu(), self.V.cpu(), self.E)

    @property
    def dim(self):
        return len(self.E)

    def toHist(self):
        return dataToHist(self.X, self.Y, self.E, self.V)

    @staticmethod
    def fromHistogram(histogram):
        edges = tuple(torch.from_numpy(a.edges) for a in histogram.axes)
        centers = tuple(torch.diff(e) / 2 + e[:-1] for e in edges)
        bin_values = torch.from_numpy(histogram.values())
        bin_vars = torch.from_numpy(histogram.variances())
        bin_values = bin_values.T
        bin_vars = bin_vars.T
        centers_grid = torch.meshgrid(*centers, indexing="xy")
        centers_grid = torch.stack(centers_grid, axis=-1)

        flat_centers = torch.flatten(centers_grid, end_dim=1)
        flat_bin_values = torch.flatten(bin_values)
        flat_bin_vars = torch.flatten(bin_vars)

        ret = DataValues(
            flat_centers,
            flat_bin_values,
            flat_bin_vars,
            edges,
        )
        return ret


# def makeRegressionData(
#     histogram,
#     mask_function=None,
#     exclude_less=None,
#     get_mask=False,
#     get_shaped_mask=False,
#     domain_mask_function=None,
#     get_window_mask=False,
# ):
#     if mask_function is None:
#         mask_function = lambda x: (torch.full_like(x[..., 0], False, dtype=torch.bool))
#
#     edges = tuple(torch.from_numpy(a.edges) for a in histogram.axes)
#     centers = tuple(torch.diff(e) / 2 + e[:-1] for e in edges)
#     bin_values = torch.from_numpy(histogram.values())
#     bin_vars = torch.from_numpy(histogram.variances())
#     if len(edges) == 2:
#         bin_values = bin_values.T
#         bin_vars = bin_vars.T
#
#     centers_grid = torch.meshgrid(*centers, indexing="xy")
#     if exclude_less:
#         domain_mask = bin_values < exclude_less
#     else:
#         domain_mask = torch.full_like(bin_values, False, dtype=torch.bool)
#
#     centers_grid = torch.stack(centers_grid, axis=-1)
#     if domain_mask_function is not None:
#         domain_mask = domain_mask | domain_mask_function(centers_grid)
#
#     print(domain_mask.shape)
#     m = mask_function(centers_grid)
#     print(m.shape)
#     centers_mask = m | domain_mask
#     flat_centers = torch.flatten(centers_grid, end_dim=1)
#     flat_bin_values = torch.flatten(bin_values)
#     flat_bin_vars = torch.flatten(bin_vars)
#     ret = DataValues(
#         flat_centers[torch.flatten(~centers_mask)],
#         flat_bin_values[torch.flatten(~centers_mask)],
#         flat_bin_vars[torch.flatten(~centers_mask)],
#         edges,
#     )
#     ret = (ret,)
#     if get_mask:
#         ret = (*ret, torch.flatten(~centers_mask))
#     if get_shaped_mask:
#         ret = (*ret, centers_mask)
#     return ret

# model.eval()
# if val is not None:
#     v = val(model)
# output = model(train_data.X)
# model.train()
# chi2 = chi2Bins(
#     output.mean, train_data.Y, train_data.V, mask=chi2mask
# )  # .item()
#
# # loss =  loss + abs(1 - chi2)
# chi2_p = chi2Bins(output.mean, train_data.Y, output.variance).item()
# s = (
#     f"Iter {i} (lr={slr:0.4f}): Loss={round(loss.item(),4)},"
#     f"X2/B={chi2.item():0.2f}, "
#     f"X2P/B={chi2_p:0.2f}"
# )
# if val is not None:
#     s += f" Val={v:0.2f}"
# for n, p in model.named_parameters():
#     x = p.flatten().round(decimals=2).tolist()
#     if not isinstance(x, list) or len(x) < 4:
#         print(f"{n} = {x}")
# ls = None
# try:
#     if hasattr(model.covar_module.base_kernel, "lengthscale"):
#         ls = model.covar_module.base_kernel.lengthscale
#     elif hasattr(model.covar_module.base_kernel.base_kernel, "lengthscale"):
#         ls = model.covar_module.base_kernel.base_kernel.lengthscale
# except Exception as e:
#     pass
#
# if ls is not None:
#     print(f"lengthscale = {ls.round(decimals=2).tolist()}")
#
# print(s)
#
# evidence = float(loss.item())
# # if chi2 < 1.05 and i > 20:
# #     break


def optimizeHyperparams(
    model,
    likelihood,
    train_data,
    iterations=200,
    lr=0.01,
):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=iterations // 1, gamma=0.1
    )
    evidence = None

    for i in range(iterations):
        optimizer.zero_grad()
        output = model(train_data.X)
        loss = -mll(output, train_data.Y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        slr = scheduler.get_last_lr()[0]

        if (i % (iterations // 20) == 0) or i == iterations - 1:
            print(f"Iter {i} (lr={slr:0.4f}): Loss={round(loss.item(),4)}")

    return model, likelihood, loss


def histToData(inhist, window_func, min_counts=1000, domain_mask_cut=None):
    train_data, window_mask, *_ = regression.makeRegressionData(
        inhist,
        window_func,
        domain_mask_function=domain_mask_cut,
        exclude_less=min_counts,
        get_mask=True,
    )
    test_data, domain_mask, shaped_mask = regression.makeRegressionData(
        inhist,
        None,
        get_mask=True,
        get_shaped_mask=True,
        domain_mask_function=domain_mask_cut,
        exclude_less=min_counts,
    )
    s = 1.0
    return train_data, test_data, domain_mask


def doCompleteRegression(
    histogram,
    model_class,
    domain_blinder,
    window_blinder,
    use_cuda=True,
    iterations=300,
    lr=0.001,
):

    all_data = DataValues.fromHistogram(histogram)
    print(all_data.X.shape)
    domain_mask = domain_blinder(all_data.X, all_data.Y)
    test_data = all_data[domain_mask]

    window_mask = window_blinder(test_data.X)
    train_data = test_data[~window_mask]

    train_transform = transformations.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)
    print(normalized_train_data.X.shape)

    if torch.cuda.is_available() and use_cuda:
        print("USING CUDA")
        train = normalized_train_data.toGpu()
        norm_test = normalized_test_data.toGpu()
    else:
        train = normalized_train_data
        norm_test = normalized_test_data

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train.V,
        learn_additional_noise=False,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )
    model = model_class(train.X, train.Y, likelihood)

    if torch.cuda.is_available() and use_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()

    model, likelihood, evidence = optimizeHyperparams(
        model,
        likelihood,
        train,
        iterations=iterations,
        lr=lr,
    )

    if torch.cuda.is_available() and use_cuda:
        model = model.cpu()
        likelihood = likelihood.cpu()

    model.eval()
    likelihood.eval()

    trained_model = TrainedModel(
        model_class=model_class,
        model_state=model.state_dict(),
        input_data=histogram,
        domain_mask=domain_mask,
        blind_mask=window_mask,
        transform=train_transform,
        metadata={},
    )
    #code.interact(local=dict(globals(), **locals()))

    return trained_model
