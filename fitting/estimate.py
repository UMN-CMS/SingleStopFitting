import argparse
import logging
import pickle as pkl
import random
from pathlib import Path

import numpy as np
import scipy
from .regression import DataValues

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import torch
from rich import print

from . import models, regression
from .blinder import GaussianWindow2D, MinYCut
from .plots import makeDiagnosticPlots
from .predictive import makePosteriorPred
from .utils import chi2Bins, dataToHist

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
    model, transform, all_data, pred_dist = regression.getPrediction(trained_model)
    pred_data = DataValues(all_data.X, pred_dist.mean, pred_dist.variance, all_data.E)

    if trained_model.blind_mask is not None:
        train_mask = trained_model.blind_mask
    else:
        train_mask = torch.full_like(all_data.Y, fill_value=True, dtype=bool)

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


def regress(
    data,
    base_save_dir,
    window=None,
    min_counts=0,
    use_cuda=False,
    window_spread=1.0,
):

    base_save_dir = Path(base_save_dir)
    base_save_dir.mkdir(exist_ok=True, parents=True)
    model = models.MyNNRBFModel2D
    trained_model = regression.doCompleteRegression(
        data,
        model,
        MinYCut(min_y=min_counts),
        window,
        iterations=20,
        use_cuda=use_cuda,
    )
    return trained_model


def estimateSingle2D(
    background_path,
    signal_path,
    signal_name,
    signal_selection,
    background_name,
    base_dir,
    window_spread=1.0,
    use_cuda=False,
    signal_injections=None,
):
    signal_injections = signal_injections or [0.0]
    base_dir = Path(base_dir)

    with open(signal_path, "rb") as f:
        signal312 = pkl.load(f)
    with open(background_path, "rb") as f:
        background = pkl.load(f)

    bkg_hist = background
    bkg_hist = bkg_hist
    a1, a2 = bkg_hist.axes
    a1_min, a1_max = a1.edges.min(), a1.edges.max()
    a2_min, a2_max = a2.edges.min(), a2.edges.max()

    signal_hist = signal312[signal_name, signal_selection]["hist"][
        hist.loc(a1_min) : hist.loc(a1_max),
        hist.loc(a2_min) : hist.loc(a2_max),
    ]

    sig_dir = base_dir  # / signal_name
    sig_dir.mkdir(exist_ok=True, parents=True)
    signal_regression_data = DataValues.fromHistogram(signal_hist)
    try:
        window = GaussianWindow2D.fromData(signal_regression_data, spread=window_spread)
    except (scipy.optimize.OptimizeWarning, RuntimeError):
        window = None
    sd = dict(
        signal_data=signal_regression_data,
        signal_hist=signal_hist,
        signal_name=signal_name,
    )
    torch.save(sd, sig_dir / "signal_data.pth")
    for r in signal_injections:
        save_dir = sig_dir / f"inject_r_{str(round(r,3)).replace('.','p')}"
        meta = {"signal_name": signal_name}
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Injecting background with signals strength {round(r,3)}")
        to_estimate = bkg_hist + r * signal_hist
        trained_model = regress(
            to_estimate,
            base_dir,
            min_counts=10,
            window=window,
            use_cuda=True,
        )
        trained_model.metadata.update({"signal_injected": r})
        # trained_model.metadata.update(add_metadata)
        torch.save(trained_model, save_dir / "bkg_estimation_result.pth")


def makeSimulatedBackground(inhist, model_class, outdir, use_cuda=True, num=10):
    trained_model = regression.doCompleteRegression(
        inhist,
        None,
        model_class,
        use_cuda=use_cuda,
        min_counts=0,
    )
    model, transform, all_data, pred_dist = regression.getPrediction(trained_model)
    poiss = torch.distributions.Poisson(torch.clamp(pred_dist.mean, min=0))

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    torch.save(trained_model, outdir / "simulated_trained_model.pth")
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
    parser.add_argument("-r", "--region", type=str, help="Region", required=True)
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=False)

    return parser.parse_args()


def main():
    mpl.use("Agg")
    mplhep.style.use("CMS")
    args = parse_arguments()

    estimateSingle2D(
        background_path=args.background,
        signal_path=args.signal,
        signal_name=args.name,
        signal_selection=args.region,
        background_name=None,
        base_dir=args.outdir,
        use_cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
