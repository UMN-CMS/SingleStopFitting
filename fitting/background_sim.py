import pickle as pkl
from .regression import DataValues
import gpytorch
import mplhep
from fitting.estimate import validate
from .utils import chi2Bins
from .predictive import makePosteriorPred
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from . import regression
from . import models
from .diagnostics import makeDiagnosticPlots
from .blinder import GaussianWindow2D, MinYCut
from .utils import dataToHist
import hist
import logging

logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float64)


def makeSimulatedBackground(
    inhist,
    outdir,
    use_cuda=True,
    asimov=False,
):
    mplhep.style.use("CMS")
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    inhist = inhist.copy(deep=True)
    v = inhist.view(flow=False).variance
    inhist.view(flow=False).variance = np.clip(v, a_min=20, a_max=None)

    trained_model = regression.doCompleteRegression(
        inhist,
        models.MyNNRBFModel2D,
        MinYCut(min_y=-1),
        None,
        use_cuda=use_cuda,
        iterations=200,
        learn_noise=False,
        lr=0.01,
        validate_function=validate,
    )

    model = regression.loadModel(trained_model)
    all_data, bm = regression.getModelingData(trained_model)
    complete_data = regression.DataValues.fromHistogram(inhist)

    init_sum = all_data.Y.sum()
    print(f"Initial total is {init_sum}")
    pred_dist = regression.getPosteriorProcess(
        model, complete_data, trained_model.transform
    )
    pred_data = DataValues(
        complete_data.X, pred_dist.mean, pred_dist.variance, complete_data.E
    )

    mask = all_data.V > 0
    global_chi2_bins = chi2Bins(pred_data.Y, complete_data.Y, complete_data.V)
    # blinded_chi2_bins = chi2Bins(all_data.Y, pred_data.Y, all_data.V, bm)
    print(f"Global Chi2/bins = {global_chi2_bins}")
    # print(f"Blinded Chi2/bins = {blinded_chi2_bins}")
    # data = {
    #     "global_chi2/bins": float(global_chi2_bins),
    #     "blinded_chi2/bins": float(global_chi2_bins),
    # }

    def saveFunc(name, fig):
        ext = "png"
        name = name.replace("(", "").replace(")", "").replace(".", "p")
        print(name)
        fig.savefig((outdir / name).with_suffix(f".{ext}"))
        plt.close(fig)

    print(f"Expcted total is {pred_dist.mean.sum()}")

    diagnostic_plots = makeDiagnosticPlots(
        pred_data, complete_data, complete_data, saveFunc
    )
    makePosteriorPred(pred_dist, complete_data, saveFunc)

    poiss = torch.distributions.Poisson(torch.clamp(pred_dist.mean, min=0))
    torch.save(trained_model, outdir / "simulated_trained_model.pth")

    fig, ax = plt.subplots()
    inhist.plot(ax=ax)
    fig.savefig(outdir / f"orig.png")
    plt.close(fig)

    for i in range(20):
        o = outdir / f"background_{i}"
        o.mkdir(exist_ok=True, parents=True)

        def saveFunc(name, fig):
            ext = "png"
            name = name.replace("(", "").replace(")", "").replace(".", "p")
            fig.savefig((o / name).with_suffix(f".{ext}"))
            plt.close(fig)

        if not asimov:
            sampled = poiss.sample()
        else:
            sampled = torch.clamp(torch.round(pred_dist.mean), min=0)

        vals = dataToHist(complete_data.X, sampled, complete_data.E, sampled)
        new_hist = inhist.copy(deep=True)
        new_hist.view(flow=True).value = 0
        new_hist.view(flow=True).variance = 0
        new_hist.view(flow=False).value = vals.values()
        new_hist.view(flow=False).variance = vals.variances()
        ratio = new_hist.sum().value / init_sum
        new_hist *= 1 / ratio

        with open(o / f"background_{i}.pkl", "wb") as f:
            pkl.dump(new_hist, f)
        fig, ax = plt.subplots()
        new_hist.plot(ax=ax)
        fig.savefig(o / f"background_{i}_plot.png")
        plt.close(fig)

        pred_data = regression.DataValues(
            complete_data.X, sampled, 100000 * torch.ones_like(sampled), all_data.E
        )

        diagnostic_plots = makeDiagnosticPlots(
            pred_data, complete_data, complete_data, saveFunc
        )


def loadHistogram(path, name, selection, x_bounds=None, y_bounds=None):
    import hist

    with open(path, "rb") as f:
        background = pkl.load(f)

    def L(l):
        if l is not None:
            return hist.loc(l)
        return None

    ret = background[name, selection]["hist"][
        L(x_bounds[0]) : L(x_bounds[1]),
        L(y_bounds[0]) : L(y_bounds[1]),
    ]

    return ret


def handleSim(args):
    h = loadHistogram(
        args.input, args.name, args.selection, args.x_bounds, args.y_bounds
    )
    if args.rebin:
        h = h[hist.rebin(args.rebin), hist.rebin(args.rebin)]
    if args.only_clip:
        outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        with open(outdir / f"background.pkl", "wb") as f:
            pkl.dump(h, f)

    else:
        makeSimulatedBackground(h, args.outdir, use_cuda=args.cuda, asimov=args.asimov)


def addSimParser(parser):
    parser.add_argument(
        "-o", "--outdir", required=True, type=str, help="Output directory"
    )

    import argparse

    def coords(s):
        try:
            x, y = map(float, s.split(","))
            return x, y
        except:
            raise argparse.ArgumentTypeError("Coordinates must be x,y")

    parser.add_argument(
        "-x", "--x-bounds", required=True, type=coords, help="Bounds for x coordinated"
    )
    parser.add_argument(
        "-y", "--y-bounds", required=True, type=coords, help="Bounds for y coordinate"
    )
    parser.add_argument("-s", "--selection", required=True, type=str, help="Selection")
    parser.add_argument("-a", "--asimov", default=False, action="store_true")
    parser.add_argument("-n", "--name", required=True, type=str, help="Name")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input")
    parser.add_argument(
        "--cuda",
        help="Use cuda",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("-r", "--rebin", default=None, type=int, help="Rebinning")
    parser.add_argument(
        "-c", "--only-clip", action="store_true", default=False, help="Only clip"
    )

    parser.set_defaults(func=handleSim)
