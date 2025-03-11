from pathlib import Path

from .regression import DataValues

import matplotlib.pyplot as plt
from rich import print

from . import regression
from .plotting.plots import makeDiagnosticPlots
from .predictive import makePosteriorPred
from .utils import chi2Bins


def plotDiagnostics(save_dir, trained_model):
    model = regression.loadModel(trained_model)
    all_data, train_mask = regression.getModelingData(trained_model)
    pred_dist = regression.getPosteriorProcess(model, all_data, trained_model.transform)

    pred_data = DataValues(all_data.X, pred_dist.mean, pred_dist.variance, all_data.E)

    global_chi2_bins = chi2Bins(pred_data.Y, all_data.Y, all_data.V)
    blinded_chi2_bins = chi2Bins(all_data.Y, pred_data.Y, all_data.V, train_mask)
    print(f"Global Chi2/bins = {global_chi2_bins}")
    print(f"Blinded Chi2/bins = {blinded_chi2_bins}")
    data = {
        "global_chi2/bins": float(global_chi2_bins),
        "blinded_chi2/bins": float(global_chi2_bins),
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    def saveFunc(name, fig):
        ext = "png"
        fig.savefig((save_dir / name).with_suffix(f".{ext}"))
        plt.close(fig)

    diagnostic_plots = makeDiagnosticPlots(
        pred_data,
        all_data,
        all_data.getMasked(~train_mask),
        saveFunc,
        mask=train_mask,
    )

    makePosteriorPred(pred_dist, all_data, saveFunc, train_mask)


def main(args):
    import torch
    out = args.outdir or Path(args.input).parent
    m = torch.load(args.input)
    plotDiagnostics(out, m)
    


def addToParser(parser):
    parser.add_argument(
        "-o", "--outdir", default=None, help="Output directory for plots"
    )

    parser.add_argument("input")
    parser.set_defaults(func=main)
    return parser
