from pathlib import Path
from fitting.utils import getScaledEigenvecs
import gpytorch

from .regression import DataValues
import torch

import matplotlib.pyplot as plt
from rich import print

from . import regression
from .plotting.plots import makeDiagnosticPlots, makeCovariancePlots, plotRaw
from .predictive import makePosteriorPred, makePValuePlots
from .utils import chi2Bins
import logging

logger = logging.getLogger(__name__)


def plotDiagnostics(save_dir, trained_model, **kwargs):
    model = regression.loadModel(trained_model)
    tm = trained_model.metadata
    coupling, mt, mx = tm["coupling"], tm["mt"], tm["mx"]
    mt, mx = float(mt), float(mx)
    all_data, train_mask = regression.getModelingData(trained_model)
    extra_noise = None
    if trained_model.learned_noise:
        extra_noise = model.likelihood.second_noise
    logger.info(f"Diagonstics: extra noise is {extra_noise}")

    pred_dist = regression.getPosteriorProcess(
        model, all_data, trained_model.transform, extra_noise=extra_noise
    )

    pred_data = DataValues(all_data.X, pred_dist.mean, pred_dist.variance, all_data.E)

    mask = all_data.V > 0
    global_chi2_bins = chi2Bins(pred_data.Y, all_data.Y, all_data.V, mask & ~train_mask)
    blinded_chi2_bins = chi2Bins(all_data.Y, pred_data.Y, all_data.V, train_mask & mask)

    # final_nlpd = gpytorch.metrics.negative_log_predictive_density(pred_dist, test_y)

    # l = model.likelihood
    # post = l(pred_dist)

    print(f"Global Chi2/bins = {global_chi2_bins}")
    print(f"Blinded Chi2/bins = {blinded_chi2_bins}")
    data = {
        "global_chi2/bins": float(global_chi2_bins),
        "blinded_chi2/bins": float(global_chi2_bins),
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    def saveFunc(name, obj):
        import json

        if isinstance(obj, dict):
            with open(save_dir / f"{name}.json", "w") as f:
                json.dump(obj, f)
        else:
            ext = "pdf"
            name = name.replace("(", "").replace(")", "").replace(".", "p")
            print(name)
            obj.savefig((save_dir / name).with_suffix(f".{ext}"))
            plt.close(obj)


    diagnostic_plots = makeDiagnosticPlots(
        pred_data,
        all_data,
        all_data.getMasked(~train_mask),
        saveFunc,
        mask=train_mask,
        **kwargs,
    )

    saveFunc("chi2_info", data)

    makePValuePlots(pred_dist, all_data, train_mask, saveFunc)
    makePosteriorPred(pred_dist, all_data, saveFunc, train_mask)
    for point in [[mt, mx / mt]]:
        makeCovariancePlots(model, trained_model.transform, all_data, point, saveFunc)


def plotCovarsForPoints(save_dir, trained_model, points):
    import matplotlib as mpl
    import mplhep

    mpl.use("Agg")
    mplhep.style.use("CMS")
    model = regression.loadModel(trained_model)
    all_data, train_mask = regression.getModelingData(trained_model)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for point in points:

        def saveFunc(name, fig):
            ext = "pdf"
            name = name.replace("(", "").replace(")", "").replace(".", "p")
            print(name)
            fig.savefig((save_dir / name).with_suffix(f".{ext}"))
            plt.close(fig)

        makeCovariancePlots(model, trained_model.transform, all_data, point, saveFunc)


def plotEigenvars(save_dir, trained_model, sig_percent=0.05):
    import matplotlib as mpl
    import mplhep
    import torch

    mpl.use("Agg")
    mplhep.style.use("CMS")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model = regression.loadModel(trained_model)
    all_data, train_mask = regression.getModelingData(trained_model)

    pred = regression.getPosteriorProcess(model, all_data, trained_model.transform)

    cov_mat = pred.covariance_matrix
    vals, vecs = getScaledEigenvecs(cov_mat)

    wanted = vals > vals[0] * sig_percent
    nz = int(torch.count_nonzero(wanted))
    print(f"There are {nz} egeinvariations at least {sig_percent} of the max ")
    good_vals, good_vecs = vals[wanted], vecs[wanted]
    for i, (va, ve) in enumerate(zip(good_vals, good_vecs)):

        def saveFunc(name, fig):
            ext = "pdf"
            name = name.replace("(", "").replace(")", "").replace(".", "p")
            print(name)
            fig.savefig((save_dir / name).with_suffix(f".{ext}"))
            plt.close(fig)

        fig, ax = plt.subplots()
        plotRaw(ax, all_data.E, all_data.X, va * ve)
        saveFunc(f"eigenvar_{i}__{round(float(va),1)}".replace(".", "p"), fig)


def main(args):
    import torch
    import matplotlib as mpl
    import mplhep

    mpl.use("Agg")
    mplhep.style.use("CMS")

    out = args.outdir or Path(args.input).parent
    m = torch.load(args.input, weights_only=False)
    plotDiagnostics(out, m)


def runEigens(args):
    import torch

    out = args.outdir or Path(args.input).parent
    m = torch.load(args.input, weights_only=False)
    plotEigenvars(out, m, args.min_frac)


def runCovars(args):
    import torch

    out = args.outdir or Path(args.input).parent
    m = torch.load(args.input, weights_only=False)
    plotCovarsForPoints(out, m, args.points)


def addDiagnosticsToParser(parser):
    parser.add_argument(
        "-o", "--outdir", default=None, help="Output directory for plots"
    )

    parser.add_argument("input")
    parser.set_defaults(func=main)
    return parser


def addCovarsToParser(parser):
    import argparse

    def coords(s):
        try:
            x, y = map(float, s.split(","))
            return x, y
        except:
            raise argparse.ArgumentTypeError("Coordinates must be x,y")

    parser.add_argument(
        "-o", "--outdir", default=None, help="Output directory for plots"
    )
    parser.add_argument(
        "-p",
        "--points",
        nargs="+",
        required=True,
        type=coords,
    )
    parser.add_argument("input")
    parser.set_defaults(func=runCovars)
    return parser


def addEigensToParser(parser):
    import argparse

    parser.add_argument(
        "-o", "--outdir", default=None, help="Output directory for plots"
    )
    parser.add_argument(
        "-m",
        "--min-frac",
        type=float,
        default=0.05,
    )
    parser.add_argument("input")
    parser.set_defaults(func=runEigens)
    return parser
