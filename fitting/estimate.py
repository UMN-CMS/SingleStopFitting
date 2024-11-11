import json
import logging
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
from .blinder import makeWindow2D, windowPlot2D
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


def histToData(inhist, window_func, min_counts=10):
    train_data, window_mask, *_ = regression.makeRegressionData(
        inhist,
        window_func,
        domain_mask_function=bumpCut,
        exclude_less=min_counts,
        get_mask=True,
    )
    test_data, domain_mask, shaped_mask = regression.makeRegressionData(
        inhist,
        None,
        get_mask=True,
        get_shaped_mask=True,
        domain_mask_function=bumpCut,
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
):

    train_data, test_data, domain_mask = histToData(inhist, window_func)
    train_transform = transformations.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)

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
    if torch.cuda.is_available() and use_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()

    def validate(model):
        X = normalized_test_data.X.cuda()
        Y = normalized_test_data.Y.cuda()
        V = normalized_test_data.V.cuda()
        if window_func is not None:
            mask = regression.getBlindedMask(test_data.X, window_func)
        else:
            mask = torch.ones_like(test_data.Y, dtype=bool)
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
        iterations=250,
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
    diagnostic_plots = makeDiagnosticPlots(
        test_pred_data,
        test_data,
        train_data,
        mask,
        inducing_points=train_transform.transform_x.iTransformData(
            model.covar_module.inducing_points
        )
        .detach()
        .numpy(),
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
    )

    torch.save(save_data, save_dir / "train_model.pth")

    with open(save_dir / "info.json", "w") as f:
        json.dump(data, f)
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

    if signal_hist:
        signal_regression_data, *_ = regression.makeRegressionData(signal_hist)
        window = makeWindow2D(signal_regression_data, frac=0.3)
        if window is None:
            print(f"COULD NOT FIND VALID WINDOW FOR SIGNAL {signal_name}")
            return
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
        fig, ax = windowPlot2D(signal_regression_data, window)
        fig.savefig(sig_dir / f"{signal_name}.pdf")

    for r in signal_injections:
        print(f"Performing estimation for signal {signal_name}.")
        print(f"Injecting background with signals strength {round(r,3)}")
        save_dir = sig_dir / f"inject_r_{str(round(r,3)).replace('.','p')}"
        save_dir.mkdir(exist_ok=True, parents=True)
        to_estimate = bkg_hist + r * signal_hist

        d = doCompleteRegression(
            to_estimate,
            window,
            save_dir=save_dir,
            model_class=models.NonStatParametric,
            # model_class=models.MyNNRBFModel,
            mean=mean,
        )
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
            # signal_injections=[0.0, 0.5, 1.0, 4.0, 16.0],
            signal_injections=[0.0],
        )

        plt.close("all")


def getHists2D(fit_region, scale=1.0):
    with open(
        "regression_results/2018_Signal312_nn_uncomp_0p67_m14_vs_mChiUncompRatio.pkl",
        "rb",
    ) as f:
        signal312 = pkl.load(f)

    with open(
        "regression_results/2018_Control_nn_uncomp_0p67_m14_vs_mChiUncompRatio.pkl",
        "rb",
    ) as f:
        control = pkl.load(f)

    print(signal312.keys())

    bkg_hist = control["Data2018", "Control"]["hist_collection"]["histogram"]
    bkg_hist = bkg_hist[fit_region] * scale

    signal_hist_names = [
        "signal_312_1000_600",
        "signal_312_1300_600",
        "signal_312_2000_1700",
        "signal_312_1500_400",
        "signal_312_1500_600",
        "signal_312_1500_100",
        "signal_312_2000_400",
        "signal_312_1000_400",
        "signal_312_1500_1000",
        "signal_312_1300_400",
        "signal_312_1200_600",
        "signal_312_1200_400",
        "signal_312_1200_700",
        "signal_312_2000_1200",
        "signal_312_1200_800",
        "signal_312_1500_900",
        "signal_312_1400_400",
        "signal_312_2000_1600",
        "signal_312_2000_900",
        "signal_312_1000_700",
    ]

    signals_to_scan = [
        (
            sn,
            signal312[sn, "Signal312"]["hist_collection"]["histogram"]["central", ...][
                fit_region
            ],
        )
        for sn in signal_hist_names
    ]

    return bkg_hist, signals_to_scan


def getHists1D(fit_regions, scale=1.0):
    with open(
        "regression_results/2018_Signal312_m14_m.pkl",
        "rb",
    ) as f:
        signal312 = pkl.load(f)

    with open(
        "regression_results/2018_Control_m14_m.pkl",
        "rb",
    ) as f:
        control = pkl.load(f)

    print(signal312.keys())

    bkg_hist = control["Data2018", "Control"]["hist_collection"]["histogram"]
    bkg_hist = bkg_hist[fit_region] * scale

    # signal_hist_names = [
    #     "signal_312_1200_400",
    #     "signal_312_1200_800",
    #     "signal_312_1500_400",
    #     "signal_312_1500_600",
    #     "signal_312_1500_1000",
    #     "signal_312_2000_900",
    #     "signal_312_2000_1200",
    # ]
    signal_hist_names = [
        "signal_312_1200_1100",
        "signal_312_1500_1400",
        "signal_312_2000_1900",
    ]
    signals_to_scan = [
        (sn, signal312[sn, "Signal312"]["hist_collection"]["histogram"]["central", ...])
        for sn in signal_hist_names
    ]
    signals_to_scan.append((None, None))
    # signals_to_scan = [(None, None)]

    return bkg_hist, signals_to_scan


def fit(path, fit_region, mc_hist=None):
    bkg_hist, signals_to_scan = getHists2D(fit_region, scale=0.1)
    doEstimationForSignals(signals_to_scan, bkg_hist, path, mean=mc_hist)


def main():
    mpl.use("Agg")
    mplhep.style.use("CMS")
    kname, kernel = "para", SK(
        gpytorch.kernels.RBFKernel(ard_num_dims=2)
    ) + models.NonStatKernel(
        dim=2, count=4
    )  # + models.NonStatKernel(dim=2, count=4)
    fit(
        "allscans/control_reduced",
        (slice(hist.loc(900), None), slice(hist.loc(0.25), None)),
    )
    # fit("allscans/control", kernel, kname, (slice(None), slice(None)))
    # fit("allscans/control_reduced", kernel, kname, (slice(hist.loc(1000), None),))
    # fit("allscans/control", kernel, kname, (slice(None)))


if __name__ == "__main__":
    main()
