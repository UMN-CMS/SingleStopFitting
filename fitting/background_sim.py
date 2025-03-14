import pickle as pkl
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from . import regression
from . import models
from .diagnostics import makeDiagnosticPlots
from .blinder import GaussianWindow2D, MinYCut
from .utils import dataToHist
import hist


def makeSimulatedBackground(inhist, outdir, use_cuda=True, rebin=1):
    inhist = inhist[hist.rebin(rebin), hist.rebin(rebin)]
    trained_model = regression.doCompleteRegression(
        inhist,
        models.MyNNRBFModel2D,
        MinYCut(min_y=3),
        None,
        use_cuda=use_cuda,
        iterations=200,
        learn_noise=False,
    )

    model = regression.loadModel(trained_model)
    data, bm = regression.getModelingData(trained_model)
    all_data = regression.DataValues.fromHistogram(inhist)

    init_sum = all_data.Y.sum()
    print(f"Initial total is {init_sum}")
    pred_dist = regression.getPosteriorProcess(model, all_data, trained_model.transform)

    poiss = torch.distributions.Poisson(torch.clamp(pred_dist.mean, min=0))
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    torch.save(trained_model, outdir / "simulated_trained_model.pth")

    fig, ax = plt.subplots()
    inhist.plot(ax=ax)
    fig.savefig(outdir / f"orig.png")
    plt.close(fig)

    for i in range(4):
        o = outdir / f"background_{i}"
        o.mkdir(exist_ok=True, parents=True)

        def saveFunc(name, fig):
            ext = "png"
            name = name.replace("(", "").replace(")", "").replace(".", "p")
            fig.savefig((o / name).with_suffix(f".{ext}"))
            plt.close(fig)

        sampled = poiss.sample()
        vals = dataToHist(all_data.X, sampled, all_data.E, sampled)
        new_hist = inhist.copy(deep=True)
        new_hist.view(flow=True).value = 0
        new_hist.view(flow=True).variance = 0
        new_hist.view(flow=False).value = vals.values()
        new_hist.view(flow=False).variance = vals.variances()
        print(f"Final total {i} is {new_hist.sum()}")
        # ratio = new_hist.sum().value / init_sum
        print(f"Ratio {ratio}")
        new_hist = (1 / ratio) * new_hist
        with open(o / f"background_{i}.pkl", "wb") as f:
            pkl.dump(new_hist, f)
        fig, ax = plt.subplots()
        new_hist.plot(ax=ax)
        fig.savefig(o / f"background_{i}_plot.png")
        plt.close(fig)

        pred_data = regression.DataValues(
            all_data.X, sampled, 100000 * torch.ones_like(sampled), all_data.E
        )

        diagnostic_plots = makeDiagnosticPlots(
            pred_data,
            all_data,
            data,
            saveFunc,
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
    hist = loadHistogram(
        args.input, args.name, args.selection, args.x_bounds, args.y_bounds
    )
    if args.only_clip:
        outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        with open(outdir / f"background.pkl", "wb") as f:
            pkl.dump(hist, f)
    else:
        makeSimulatedBackground(hist, args.outdir, use_cuda=args.use_cuda)


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
    parser.add_argument("-n", "--name", required=True, type=str, help="Name")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input")
    parser.add_argument("-u", "--use-cuda", action="store_true", default=True, help="")
    parser.add_argument(
        "-c", "--only-clip", action="store_true", default=False, help="Only clip"
    )

    parser.set_defaults(func=handleSim)
