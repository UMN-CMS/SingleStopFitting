import argparse
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
from .utils import chi2Bins, dataToHist, modelToPredMVN

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


def diagnostics(save_dir, trained_model):
    all_data, pred_dist = regression.getPrediction(trained_model)
    pred_data = regression.DataValues(
        all_data.X, pred_dist.mean, pred_dist.variance, all_data.E
    )

    train_mask = trained_model.blind_mask

    global_chi2_bins = chi2Bins(pred_data.Y, all_data.Y, all_data.V)
    # good_bin_mask = all_data.Y > min_counts
    blinded_chi2_bins = chi2Bins(all_data.Y, pred_data.Y, all_data.V, train_mask)
    print(f"Global Chi2/bins = {global_chi2_bins}")
    print(f"Blinded Chi2/bins = {blinded_chi2_bins}")
    data = {
        "global_chi2/bins": float(global_chi2_bins),
        "blinded_chi2/bins": float(global_chi2_bins),
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    diagnostic_plots = makeDiagnosticPlots(
        pred_data,
        all_data,
        all_data.getMasked(~train_mask),
        train_mask,
    )
    saveDiagnosticPlots(diagnostic_plots, save_dir)
    p, d = makePosteriorPred(pred_dist, all_data, train_mask)
    saveDiagnosticPlots(p, save_dir)


def doRegressionForSignal(
    signal_name,
    signal_hist,
    bkg_hist,
    save_dir,
    signal_injections=None,
    mean=None,
    use_cuda=False,
):
    print(f"Signal Name: {signal_name}")
    logging.info(f"Signal is: {signal_name}")

    inducing_ratio = 4
    signal_injections = signal_injections or [0.0, 1.0, 4.0, 16.0]

    dim = len(bkg_hist.axes)

    if signal_hist:
        signal_regression_data, *_ = regression.makeRegressionData(signal_hist)
        try:
            if dim == 2:
                window = makeWindow2D(signal_regression_data, spread=1.0)
                if torch.any(window.center < 0.0) or torch.any(window.center > 1.0):
                    window = None
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
    # if sig_dir.exists():
    #       print(f"Already have {sig_dir}")
    sig_dir.mkdir(exist_ok=True, parents=True)
    if signal_hist:
        torch.save(sd, sig_dir / "signal_data.pth")
        if dim == 1:
            fig, ax = windowPlot1D(signal_regression_data, window)
        else:
            window_plots = windowPlots2D(signal_regression_data, window)
        for name, (fig, _) in window_plots.items():
            fig.savefig(sig_dir / f"{name}.pdf")
    if window is None:
        print(f"COULD NOT FIND VALID WINDOW FOR SIGNAL {signal_name}")
        return

    if dim == 1:
        model = models.NonStatParametric1D
    else:
        model = models.NonStatParametric2D

    for r in signal_injections:
        print(f"Performing estimation for signal {signal_name}.")
        save_dir = sig_dir / f"inject_r_{str(round(r,3)).replace('.','p')}"
        save_dir.mkdir(exist_ok=True, parents=True)

        if signal_hist is not None:
            print(f"Injecting background with signals strength {round(r,3)}")
            to_estimate = bkg_hist + r * signal_hist
        else:
            to_estimate = bkg_hist

        trained_model = regression.doCompleteRegression(
            to_estimate,
            window,
            model_class=model,
            mean=mean,
            domain_mask_function=None if dim == 2 else None,
            min_counts=10,
            use_cuda=use_cuda,
        )
        diagnostics(save_dir, trained_model)
        trained_model.metadata.update({"signal_injected": r})
        torch.save(trained_model, save_dir / "bkg_estimation_result.pth")
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
            # signal_injections=[0.0, 1.0, 4.0, 16.0],
            signal_injections=[0.0, 1.0],
            # signal_injections=[0.0],
        )

        plt.close("all")


def estimateSingle2D(
    background_path,
    signal_path,
    fit_region,
    signal_name,
    signal_selection,
    background_name,
    background_selection,
    base_dir,
    use_cuda=False,
):
    base_dir = Path(base_dir)
    with open(signal_path, "rb") as f:
        signal312 = pkl.load(f)

    with open(background_path, "rb") as f:
        background = pkl.load(f)

    bkg_hist = background
    bkg_hist = bkg_hist[hist.loc(fit_region[0]) :, hist.loc(fit_region[1]) :]
    signal_hist = signal312[signal_name, signal_selection]["hist_collection"][
        "histogram"
    ]["central", ...][hist.loc(fit_region[0]) :, hist.loc(fit_region[1]) :]

    doRegressionForSignal(
        signal_name,
        signal_hist,
        bkg_hist,
        base_dir,
        signal_injections=[0.0, 1.0],
        use_cuda=use_cuda,
    )
    plt.close("all")



def makeSimulatedBackground(inhist, model_class, outdir, num=10):
    trained_model = regression.doCompleteRegression(
        inhist, None, model_class, min_counts=0
    )
    all_data, pred_dist = regression.getPrediction(trained_model)
    poiss = torch.distributions.Poisson(torch.clamp(pred_dist.mean, min=0))

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    for i in range(num):
        sampled = poiss.sample()
        vals = dataToHist(all_data.X, sampled, all_data.E, sampled)
        new_hist = inhist.copy(deep=True)
        new_hist.view(flow=True).value = 0
        new_hist.view(flow=True).variance = 0
        new_hist.view(flow=False).value = vals.values()
        new_hist.view(flow=False).variance = vals.variances()
        with open(outdir / f"background_{i}.pkl", "wb") as f:
            pkl.dump(new_hist, f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Do a background estimatation")
    parser.add_argument(
        "-o", "--outdir", type=str, help="Output directory for estimate", required=True
    )
    parser.add_argument(
        "-b",
        "--background",
        type=str,
        help="Input path for the background",
        required=True,
    )
    parser.add_argument(
        "-s", "--signal", type=str, help="Input path for the signal", required=True
    )
    parser.add_argument(
        "-n", "--name", type=str, help="Name for the signal", required=True
    )
    parser.add_argument(
        "-l",
        "--lower-bounds",
        type=float,
        help="Lower bounds for the histgram",
        nargs="*",
        required=False,
    )
    parser.add_argument("-r", "--region", type=str, help="Region", required=True)
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=False)

    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        help="dimenstion",
        default=2,
    )

    return parser.parse_args()


def main():
    # import warnings
    # warnings.filterwarnings('error')
    mpl.use("Agg")
    mplhep.style.use("CMS")
    # region_2d = (slice(hist.loc(925), None), slice(hist.loc(0.25), None))
    # region_1d = (slice(hist.loc(500), None),)
    args = parse_arguments()
    print(args)

    if args.dimension == 2:
        estimateSingle2D(
            background_path=args.background,
            signal_path=args.signal,
            fit_region=args.lower_bounds,
            signal_name=args.name,
            signal_selection=args.region,
            background_name=None,
            background_selection=args.region,
            base_dir=args.outdir,
            use_cuda=args.cuda,
        )

if __name__ == "__main__":
    main()
