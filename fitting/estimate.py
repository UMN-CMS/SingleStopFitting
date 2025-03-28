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
    learning_rate=0.02,
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
        learn_noise=False,
        lr=learning_rate,
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
    blinding_signal=True,
    use_cuda=False,
    signal_injections=None,
    learning_rate=0.02,
    rebin_signal=1,
    rebin_background=1,
    min_base_variance=None,
    use_other_model=None,
):
    signal_injections = signal_injections or [0.0, 1.0, 4.0, 16.0]
    base_dir = Path(base_dir)

    with open(signal_path, "rb") as f:
        signal_file = pkl.load(f)
    with open(background_path, "rb") as f:
        background = pkl.load(f)
    bkg_hist = background
    if min_base_variance:
        import numpy as np
        bkg_hist = bkg_hist.copy(deep=True)
        v = bkg_hist.view(flow=False).variance
        bkg_hist.view(flow=False).variance = np.clip(v, a_min=5, a_max=None)

    a1, a2 = bkg_hist.axes
    a1_min, a1_max = a1.edges.min(), a1.edges.max()
    a2_min, a2_max = a2.edges.min(), a2.edges.max()
    signal_hist_with_flow = signal_file[signal_name, signal_selection]["hist"]


    bkg_hist = bkg_hist[hist.rebin(rebin_background), hist.rebin(rebin_background)]


    signal_hist = signal_file[signal_name, signal_selection]["hist"][
        hist.loc(a1_min) : hist.loc(a1_max),
        hist.loc(a2_min) : hist.loc(a2_max),
    ]

    print(bkg_hist)
    print(signal_hist)

    signal_hist = signal_hist[hist.rebin(rebin_signal), hist.rebin(rebin_signal)]
    # 
    # signal_hist = signal_hist_with_flow.copy(deep=True)
    # signal_hist.view(flow=True).value = signal_hist_with_flow.values(flow=False)
    # signal_hist.view(flow=True).variance = signal_hist_with_flow.variances(flow=False)

    sig_dir = base_dir  # / signal_name
    sig_dir.mkdir(exist_ok=True, parents=True)
    signal_regression_data = DataValues.fromHistogram(signal_hist)
    if blinding_signal:
        try:
            window = GaussianWindow2D.fromData(
                signal_regression_data, spread=window_spread
            )
        except (scipy.optimize.OptimizeWarning, RuntimeError) as e:
            raise e
            window = None
    else:
        logger.warn(f"Could not find a window for signal {signal_name}")
        window = None
    sd = dict(
        signal_data=signal_regression_data,
        signal_hist=signal_hist,
        signal_name=signal_name,
    )

    print(bkg_hist)
    print(signal_hist)

    def saveFunc(name, fig):
        ext = "png"
        fig.savefig((sig_dir / name).with_suffix(f".{ext}"))
        plt.close(fig)

    windowPlots2D(signal_regression_data, window, saveFunc)
    torch.save(sd, sig_dir / "signal_data.pth")
    for r in signal_injections:
        print(window)
        save_dir = sig_dir / f"inject_r_{str(round(r,3)).replace('.','p')}"
        meta = {"signal_name": signal_name}
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Injecting background with signals strength {round(r,3)}")
        to_estimate = bkg_hist + r * signal_hist
        if use_other_model:
            trained_model = regression.updateModelNewData(
                use_other_model,
                to_estimate,
                MinYCut(min_y=0),
                window,
            )

        else:
            trained_model = regress(
                to_estimate,
                base_dir,
                min_counts=-1,
                window=window,
                use_cuda=True,
                learning_rate=learning_rate,
            )
        trained_model.metadata.update({"signal_injected": r})
        # trained_model.metadata.update(add_metadata)
        torch.save(trained_model, save_dir / "bkg_estimation_result.pth")
        plotDiagnostics(save_dir, trained_model)


def main(args):
    mpl.use("Agg")
    mplhep.style.use("CMS")
    if not args.blind_signal:
        logger.warn(f"Not blinding signal window")

    other_model = None
    if args.use_other_model:
        other_model_data = torch.load(args.use_other_model)
        other_model = regression.loadModel(other_model_data)

    estimateSingle2D(
        background_path=args.background,
        signal_path=args.signal,
        signal_name=args.name,
        signal_selection=args.region,
        background_name=None,
        base_dir=args.outdir,
        use_cuda=args.cuda,
        window_spread=1.0,
        learning_rate=args.learning_rate,
        rebin_signal=args.rebin_signal,
        rebin_background=args.rebin_background,
        blinding_signal=args.blind_signal,
        signal_injections=[0.0, 1.0, 4.0, 16.0],
        min_base_variance=5,
        use_other_model=other_model,
    )


def addToParser(parser):
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
    parser.add_argument("--rebin-signal", default=1, type=int, help="Rebinning")
    parser.add_argument("--rebin-background", default=1, type=int, help="Rebinning")
    parser.add_argument("-r", "--region", type=str, help="Region", required=True)
    parser.add_argument("-l", "--learning-rate", type=float, default=0.02)
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=False)
    parser.add_argument("--use-other-model", type=str)
    parser.add_argument(
        "--blind-signal", default=True, action=argparse.BooleanOptionalAction
    )

    parser.set_defaults(func=main)
    return parser
