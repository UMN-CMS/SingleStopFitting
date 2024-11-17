import json
import logging
import scipy
import pickle as pkl
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
from .utils import chi2Bins, modelToPredMVN

torch.set_default_dtype(torch.float64)
torch.manual_seed(12345)
random.seed(12345)
np.random.seed(12345)


logger = logging.getLogger(__name__)


def saveDiagnosticPlots(plots, save_dir):
    for name, (fig, ax) in plots.items():
        o = (save_dir / name).with_suffix(".pdf")
        fig.savefig(o)
        plt.close(fig)


def bumpCut(X, Y):
    m = Y > (1 - 200 / X)

    # m = torch.zeros_like(X, dtype=bool)
    return m


def makePostData(model, likelihood, data, base_data, slope=None, intercept=None):
    pred_dist = modelToPredMVN(
        model, likelihood, data, slope=slope, intercept=intercept
    )

    return regression.DataValues(
        base_data.X, pred_dist.mean, pred_dist.variance, base_data.E
    )


def histToData(inhist, window_func, min_counts=10, domain_mask_cut=None):
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
    save_dir="plots",
    mean=None,
    just_model=False,
    use_cuda=True,
    min_counts=0,
    domain_mask_function=None,
):

    train_data, test_data, domain_mask = histToData(
        inhist, window_func, domain_mask_cut=domain_mask_function
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

    lr = 0.05

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
            mask = regression.getBlindedMask(test_data.X, window_func)
        else:
            mask = torch.ones_like(test_data.Y, dtype=bool)
        if use_cuda:
            mask = mask.cuda()
        output = model(X)
        bpred_mean = output.mean
        chi2 = chi2Bins(Y, bpred_mean, V, mask)
        # chi2 = chi2Bins(Y, bpred_mean, output.variance, mask)
        return chi2

    model, likelihood, evidence = regression.optimizeHyperparams(
        model,
        likelihood,
        train,
        bar=False,
        iterations=100,
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

    slope = train_transform.transform_y.slope
    intercept = train_transform.transform_y.intercept

    # raw_pred_dist = modelToPredMVN(model, likelihood, normalized_train_data)
    pred_dist = modelToPredMVN(
        model, likelihood, normalized_test_data, slope, intercept
    )

    train_pred_data = makePostData(
        model, likelihood, normalized_train_data, train_data, slope, intercept
    )
    test_pred_data = makePostData(
        model, likelihood, normalized_test_data, test_data, slope, intercept
    )

    good_bin_mask = test_data.Y > min_counts
    global_chi2_bins = chi2Bins(
        test_pred_data.Y, test_data.Y, test_data.V, good_bin_mask
    )
    good_bin_mask = train_data.Y > min_counts
    blinded_chi2_bins = chi2Bins(
        train_pred_data.Y, train_data.Y, train_data.V, good_bin_mask
    )
    print(f"Global Chi2/bins = {global_chi2_bins}")
    print(f"Unblinded Chi2/bins = {blinded_chi2_bins}")
    data = {
        "evidence": float(evidence),
        "global_chi2/bins": float(global_chi2_bins),
        "model_string": str(model),
    }

    if window_func:
        mask = regression.getBlindedMask(test_pred_data.X, window_func)
        bpred_mean = test_pred_data.Y
        obs_mean = test_data.Y
        obs_var = test_data.V
        chi2 = chi2Bins(obs_mean, bpred_mean, obs_var, mask)
        avg_chi = torch.sum(
            (obs_mean - bpred_mean) / torch.sqrt(obs_var)
        ) / obs_mean.size(0)
        abs_avg_chi = torch.sum(
            torch.abs(obs_mean - bpred_mean) / torch.sqrt(obs_var)
        ) / obs_mean.size(0)
        data.update(
            {
                "chi2_blinded": float(chi2),
                "avg_chi_blinded": float(avg_chi),
                "abs_avg_chi_blinded": float(abs_avg_chi),
            }
        )
        print(f"Blinded Chi^2/bins = {float(chi2)}")
        print(f"Blinded Chi/bins = {float(avg_chi)}")
        print(f"Blinded Abs(Chi)/bins = {float(abs_avg_chi)}")
    else:
        mask = None

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    if hasattr(model.covar_module, "inducing_points"):
        ind = (
            train_transform.transform_x.iTransformData(
                model.covar_module.inducing_points
            )
            .detach()
            .numpy()
        )
    else:
        ind = None

    diagnostic_plots = makeDiagnosticPlots(
        test_pred_data, test_data, train_data, mask, inducing_points=ind
    )
    p, d = makePosteriorPred(pred_dist, test_data, mask)
    diagnostic_plots.update(p)
    data.update(d)
    try:
        diagnostic_plots.update(makeNNPlots(model, test_data))
        print("Saving NN")
    except AttributeError as e:
        pass
    saveDiagnosticPlots(diagnostic_plots, save_dir)

    model_dict = model.state_dict()
    save_data = dict(
        model_name=type(model).__name__,
        input_data=inhist,
        domain_mask=domain_mask,
        blind_mask=mask,
        transform=train_transform,
        model_dict=model.state_dict(),
        metadata=data,
    )

    return save_data


def doRegressionForSignal(
    signal_name,
    signal_hist,
    bkg_hist,
    save_dir,
    signal_injections=None,
    mean=None,
):
    logging.info(f"Signal is: {signal_name}")

    inducing_ratio = 4
    signal_injections = signal_injections or [0.0, 1.0, 4.0, 16.0]

    dim = len(bkg_hist.axes)

    if signal_hist:
        signal_regression_data, *_ = regression.makeRegressionData(signal_hist)
        try:
            if dim == 2:
                window = makeWindow2D(signal_regression_data, spread=1.0)
            elif dim == 1:
                window = makeWindow1D(signal_regression_data, spread=1.3)
        except (scipy.optimize.OptimizeWarning, RuntimeError):
            window = None


        sd = dict(
            signal_data=signal_regression_data,
            signal_hist=signal_hist,
            signal_name=signal_name,
        )
    else:
        window = None

    sig_dir = save_dir / (signal_name if signal_name else "NoSignal")
    sig_dir.mkdir(exist_ok=True, parents=True)
    if signal_hist:
        torch.save(sd, sig_dir / "signal_data.pth")
        if dim == 1:
            fig, ax = windowPlot1D(signal_regression_data, window)
        else:
            window_plots = windowPlots2D(signal_regression_data, window)
        for name, (fig,_) in window_plots.items():
            fig.savefig(sig_dir / f"{name}.pdf")
    if window is None:
        print(f"COULD NOT FIND VALID WINDOW FOR SIGNAL {signal_name}")
        return

    if dim == 1:
        model = models.NonStatParametric1D

        # model = models.MyNNRBFModel1D
    else:
        model = models.NonStatParametric2D
        model = models.MyNNRBFModel2D
        model = models.MyRBF2D

    print(model)

    for r in signal_injections:
        print(f"Performing estimation for signal {signal_name}.")
        save_dir = sig_dir / f"inject_r_{str(round(r,3)).replace('.','p')}"
        save_dir.mkdir(exist_ok=True, parents=True)

        if signal_hist is not None:
            print(f"Injecting background with signals strength {round(r,3)}")
            to_estimate = bkg_hist + r * signal_hist
        else:
            to_estimate = bkg_hist

        data = doCompleteRegression(
            to_estimate,
            window,
            save_dir=save_dir,
            model_class=model,
            mean=mean,
            domain_mask_function=None if dim == 2 else None,
        )
        data["metadata"].update({"signal_injected": r})
        torch.save(data, save_dir / "bkg_estimation_result.pth")
        plt.close("all")


def doEstimationForSignals(
    signals,
    bkg_hist,
    base_dir,
    signal_injections=None,
    mean=None,
):
    base_dir = Path(base_dir)
    for signal_name, signal_hist in signals:
        doRegressionForSignal(
            signal_name,
            signal_hist,
            bkg_hist,
            base_dir,
            mean=mean,
            #signal_injections=[0.0, 1.0, 4.0, 16.0],
            signal_injections=[0.0],
            # signal_injections=[0.0],
        )

        plt.close("all")


def getHists1D(
    background_path,
    signal_path,
    signals=None,
    fit_region=None,
    scale=1.0,
    signal_selection="Signal312",
    background_name="Data2018",
    background_selection="Control",
):
    with open(signal_path, "rb") as f:
        signal312 = pkl.load(f)

    with open(background_path, "rb") as f:
        control = pkl.load(f)

    bkg_hist = control[background_name, background_selection]["hist_collection"][
        "histogram"
    ][:, sum]
    bkg_hist = bkg_hist[fit_region] * scale

    signals = signals or [x[0] for x in signal312.keys() if "signal" in x]
    signals=reversed(signals)
    signals_to_scan = [
        (
            s,
            signal312[s, signal_selection]["hist_collection"]["histogram"][
                "central", ...
            ][:, sum][fit_region],
        )
        for s in signals
    ]
    return bkg_hist, signals_to_scan


def getHists2D(
    background_path,
    signal_path,
    signals=None,
    fit_region=None,
    scale=1.0,
    signal_selection="Signal312",
    background_name="Data2018",
    background_selection="Control",
):
    with open(signal_path, "rb") as f:
        signal312 = pkl.load(f)

    with open(background_path, "rb") as f:
        control = pkl.load(f)

    print(control.keys())
    bkg_hist = control[background_name, background_selection]["hist_collection"][
        "histogram"
    ]["central", ...]
    print(fit_region)
    bkg_hist = bkg_hist[fit_region] * scale

    print(signal312.keys())
    signals = signals or [x[0] for x in signal312.keys() if "signal" in x[0]]
    signals=reversed(signals)
    signals_to_scan = [
        (
            s,
            signal312[s, signal_selection]["hist_collection"]["histogram"][
                "central", ...
            ][fit_region],
        )
        for s in signals
    ]
    return bkg_hist, signals_to_scan


def fit(path, fit_region, mc_hist=None):
    bkg_hist, signals_to_scan = getHists2D(fit_region, scale=0.1)
    # bkg_hist, signals_to_scan = getHists1D(fit_region, scale=0.1)
    doEstimationForSignals(signals_to_scan, bkg_hist, path, mean=mc_hist)


def createEstimationGrid(
    background_path,
    signal_path,
    output_dir,
    dim,
    fit_region=None,
    signal_names=None,
    background_scale=1.0,
):
    output_dir = Path(output_dir)
    if dim == 1:
        bkg_hist, signals_to_scan = getHists1D(
            background_path,
            signal_path,
            fit_region=fit_region,
            signals=signal_names,
            scale=background_scale,
            signal_selection="Signal312",
            background_selection="Signal312",
            background_name="QCDInclusive2018",
        )
    elif dim == 2:
        bkg_hist, signals_to_scan = getHists2D(
            background_path,
            signal_path,
            fit_region=fit_region,
            signals=signal_names,
            scale=background_scale,
            signal_selection="Signal312",
            background_selection="Signal312",
            background_name="QCDInclusive2018",
        )

    doEstimationForSignals(signals_to_scan, bkg_hist, output_dir)


def main():
    # import warnings 
    # warnings.filterwarnings('error')
    mpl.use("Agg")
    mplhep.style.use("CMS")

    region_2d = (slice(hist.loc(900), None), slice(hist.loc(0.25), None))
    region_1d = (slice(hist.loc(500), None),)

    createEstimationGrid(
        "regression_results/2018_Signal312_nn_uncomp_0p67_m14_vs_mChiUncompRatio.pkl",
        "regression_results/2018_Signal312_nn_uncomp_0p67_m14_vs_mChiUncompRatio.pkl",
        "gaussian_window_results",
        2,
        # signal_names=["signal_312_1000_600"],
        fit_region=region_2d,
        background_scale=1.0,
    )
    #
    #     "allscans/control_reduced",
    # )
    # fit("allscans/control", kernel, kname, (slice(None), slice(None)))
    # fit("allscans/control_reduced_1d", (slice(hist.loc(500), None),))
    # fit("allscans/control", kernel, kname, (slice(None)))


if __name__ == "__main__":
    main()
