import argparse
import logging
import pickle as pkl
from pathlib import Path
from .diagnostics import plotDiagnostics

import scipy
from .regression import DataValues

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import torch

from . import models, regression
from .blinder import GaussianWindow2D, MinYCut
from .plotting.plots import windowPlots2D

torch.set_default_dtype(torch.float64)


logger = logging.getLogger(__name__)


def saveDiagnosticPlots(plots, save_dir):
    for name, (fig, ax) in plots.items():
        o = (save_dir / name).with_suffix(".pdf")
        fig.savefig(o)
        plt.close(fig)


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
        iterations=200,
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

    def saveFunc(name, fig):
        ext = "png"
        fig.savefig((sig_dir / name).with_suffix(f".{ext}"))
        plt.close(fig)

    windowPlots2D(signal_regression_data, window, saveFunc)
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
        plotDiagnostics(save_dir, trained_model)


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
