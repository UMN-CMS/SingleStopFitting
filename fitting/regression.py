import contextlib
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
from .blinder import makeWindow1D, makeWindow2D, windowPlot1D, windowPlots2D
from .plots import makeDiagnosticPlots, makeNNPlots
from .predictive import makePosteriorPred
from .utils import chi2Bins, modelToPredMVN, dataToHist

DataValues = namedtuple("DataValues", "X Y V E")


@dataclass
class TrainedModel:
    model_name: str
    model_dict: dict

    input_data: hist.Hist
    domain_mask: torch.Tensor
    blind_mask: torch.Tensor
    transform: torch.Tensor
    metadata: dict


def getPrediction(bkg_data, other_data=None):
    hist = bkg_data.input_data
    model_class = bkg_data.model_class
    raw_regression_data, *_ = makeRegressionData(hist)
    bm = bkg_data.blind_mask
    dm = bkg_data.domain_mask

    if other_data is None:
        all_data = raw_regression_data.getMasked(dm)
    else:
        all_data = other_data

    if bm is None:
        bm = torch.zeros_like(all_data.Y, dtype=bool)

    blinded_data = all_data.getMasked(~bm)

    transform = transformations.getNormalizationTransform(blinded_data)

    normalized_blinded_data = transform.transform(blinded_data)
    normalized_all_data = transform.transform(all_data)

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=normalized_blinded_data.V,
        learn_additional_noise=False,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )

    model = model_class(
        normalized_blinded_data.X,
        normalized_blinded_data.Y,
        likelihood,
        num_inducing=bkg_data.model_dict["covar_module.inducing_points"].size(0),
    )

    model.load_state_dict(bkg_data.model_dict)

    model.eval()
    likelihood.eval()


    pred_dist = modelToPredMVN(
        model,
        likelihood,
        normalized_all_data,
        slope=transform.transform_y.slope,
        intercept=transform.transform_y.intercept,
    )
    pred_data = DataValues(all_data.X, pred_dist.mean, pred_dist.variance, all_data.E)
    good_bin_mask = all_data.Y > 10
    global_chi2_bins = chi2Bins(pred_data.Y, all_data.Y, all_data.V, good_bin_mask)
    blinded_chi2_bins = chi2Bins(pred_data.Y, all_data.Y, all_data.V, bm)
    print(global_chi2_bins)

    return model, transform, all_data, pred_dist


@dataclass
class DataValues:
    X: torch.Tensor
    Y: torch.Tensor
    V: torch.Tensor
    E: torch.Tensor

    def getMasked(self, mask):
        return DataValues(self.X[mask], self.Y[mask], self.V[mask], self.E)

    def toGpu(self):
        return DataValues(self.X.cuda(), self.Y.cuda(), self.V.cuda(), self.E)

    def fromGpu(self):
        return DataValues(self.X.cpu(), self.Y.cpu(), self.V.cpu(), self.E)

    @property
    def dim(self):
        return len(self.E)

    def toHist(self):
        return dataToHist(self.X, self.Y, self.E, self.V)


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
        mask_function = lambda x: (torch.full_like(x[..., 0], False, dtype=torch.bool))

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
    if domain_mask_function is not None:
        domain_mask = domain_mask | domain_mask_function(centers_grid)

    print(domain_mask.shape)
    m = mask_function(centers_grid)
    print(m.shape)
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
    iterations=200,
    bar=True,
    lr=0.01,
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


def getBlindedMask(inputs, mask_func):
    mask = mask_func(inputs)
    return mask


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
    inhist,
    window_func,
    model_class=None,
    mean=None,
    just_model=False,
    use_cuda=True,
    min_counts=10,
    domain_mask_function=None,
):

    if isinstance(inhist, DataValues):
        test_data = inhist
        train_data = inhist.getMasked(
            getBlindedMask(train_data.X, window_func)
        )
        domain_mask = torch.ones_like(train_data.Y, dtype=bool)
    else:
        train_data, test_data, domain_mask = histToData(
            inhist,
            window_func,
            domain_mask_cut=domain_mask_function,
            min_counts=min_counts,
        )
    train_transform = transformations.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)
    print(normalized_train_data.X.shape)
    print(normalized_test_data.X.shape)

    if torch.cuda.is_available() and use_cuda:
        train = normalized_train_data.toGpu()
        norm_test = normalized_test_data.toGpu()
        print("USING CUDA")
    else:
        train = normalized_train_data
        norm_test = normalized_test_data


    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train.V,
        learn_additional_noise=False,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )
    model = model_class(train.X, train.Y, likelihood)
    print(model)
    if torch.cuda.is_available() and use_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()

    def validate(model):
        if use_cuda:
            X = normalized_test_data.X.cuda()
            Y = normalized_test_data.Y.cuda()
            V = normalized_test_data.V.cuda()
        else:
            X = normalized_test_data.X
            Y = normalized_test_data.Y
            V = normalized_test_data.V
        if window_func is not None:
            mask = getBlindedMask(test_data.X, window_func)
        else:
            mask = torch.ones_like(test_data.Y, dtype=bool)
        if use_cuda:
            mask = mask.cuda()
        output = model(X)
        bpred_mean = output.mean
        chi2 = chi2Bins(Y, bpred_mean, V, mask)
        # chi2 = chi2Bins(Y, bpred_mean, output.variance, mask)
        return chi2

    lr = 0.0025
    model, likelihood, evidence = optimizeHyperparams(
        model,
        likelihood,
        train,
        bar=False,
        iterations=300,
        lr=lr,
        get_evidence=True,
        chi2mask=train_data.Y > min_counts,
        val=validate,
    )

    if torch.cuda.is_available() and use_cuda:
        model = model.cpu()
        likelihood = likelihood.cpu()

    model.eval()
    likelihood.eval()

    if window_func:
        mask = getBlindedMask(test_data.X, window_func)
    else:
        mask = None

    model_dict = model.state_dict()

    save_data = TrainedModel(
        model_class=model_class,
        input_data=inhist,
        domain_mask=domain_mask,
        blind_mask=mask,
        transform=train_transform,
        model_dict=model.state_dict(),
        metadata={},
    )

    return save_data
