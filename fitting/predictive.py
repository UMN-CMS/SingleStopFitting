import numpy as np

import json
import matplotlib.pyplot as plt
from pathlib import Path
import mplhep
import pyro
import pyro.distributions as pyrod
import pyro.infer as pyroi
import fitting.regression as regression
import torch
from matplotlib.patches import Polygon

from .plot_tools import getPolyFromSquares, makeSquares
from .plots import plotRaw
from .utils import chi2Bins


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "16%": v.kthvalue(int(len(v) * 0.16), dim=0)[0],
            "84%": v.kthvalue(int(len(v) * 0.84), dim=0)[0],
        }
    return site_stats


def statModel(bkg_mvn, observed=None):
    background = pyro.sample("background", bkg_mvn)
    with pyro.plate("bins", bkg_mvn.mean.shape[0]):
        return pyro.sample(
            "observed", pyrod.Poisson(torch.clamp(background, 0)), obs=observed
        )


def getPosteriorPred(bkg_mvn, num_samples=800):
    predictive = pyroi.Predictive(
        statModel,
        num_samples=num_samples,
    )

    pred = predictive(bkg_mvn)
    return pred


def makePosteriorPred(
    bkg_mvn,
    test_data,
    save_func,
    mask=None,
):
    pred = getPosteriorPred(bkg_mvn)
    summ = summary(pred)

    stat_pulls = (test_data.Y - bkg_mvn.mean) / torch.sqrt(test_data.V)
    post_pulls = (test_data.Y - bkg_mvn.mean) / summ["observed"]["std"]
    pred_only_pulls = (test_data.Y - bkg_mvn.mean) / torch.sqrt(bkg_mvn.variance)

    global_chi2_bins = chi2Bins(bkg_mvn.mean, test_data.Y, test_data.V)
    blinded_chi2_bins = chi2Bins(bkg_mvn.mean, test_data.Y, test_data.V, mask)

    d = test_data.dim

    def addWindow(ax):
        if mask is None or not torch.any(mask):
            return
        else:
            squares = makeSquares(test_data.X[mask], test_data.E)
            points = getPolyFromSquares(squares)
            poly = Polygon(points, edgecolor="green", linewidth=3, fill=False)
            ax.add_patch(poly)

    def addChi2(ax):
        mplhep.cms.text(
            f"$\\chi^2 Global = {global_chi2_bins:0.2f}$\n$\\chi^2 Blind = {blinded_chi2_bins:0.2f}$",
            loc=3,
            fontsize=20,
        )

    fig, ax = plt.subplots(layout="tight")
    ax.set_title("Relative Uncertainty In Pred")
    f = plotRaw(
        ax,
        test_data.E,
        test_data.X,
        torch.sqrt((bkg_mvn.variance)) / bkg_mvn.mean,
    )
    addWindow(ax)
    addChi2(ax)
    save_func("post_relative_uncertainty", fig)

    fig, ax = plt.subplots(layout="tight")
    ax.set_title("Relative Uncertainty In Posterior")
    f = plotRaw(
        ax,
        test_data.E,
        test_data.X,
        summ["observed"]["std"] / bkg_mvn.mean,
    )
    addWindow(ax)
    save_func("post_posterior_relative_uncertainty", fig)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(
        ax, test_data.E, test_data.X, pred_only_pulls, cmap="coolwarm", cmin=-3, cmax=3
    )
    ax.set_title("Pull Latent Only")
    addWindow(ax)
    addChi2(ax)
    save_func("post_pull_latent", fig)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(
        ax, test_data.E, test_data.X, stat_pulls, cmap="coolwarm", cmin=-3, cmax=3
    )
    ax.set_title("Pull Statistical")
    addWindow(ax)
    addChi2(ax)
    save_func("post_pull_statistical", fig)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(
        ax, test_data.E, test_data.X, post_pulls, cmap="coolwarm", cmin=-3, cmax=3
    )
    addWindow(ax)
    ax.set_title("Pull Posterior")
    save_func("post_pull_posterior", fig)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(
        ax,
        test_data.E,
        test_data.X,
        stat_pulls - post_pulls,
        cmap="coolwarm",
    )
    ax.set_title("Pull Stat - Pull Posterior")
    save_func("post_pull_diff", fig)

    fig, ax = plt.subplots(layout="tight")
    p = post_pulls
    ax.hist(p, bins=np.linspace(-5.0, 5.0, 21), density=True)
    ax.set_title("Predictive Pull Distribution -- Full Plane")
    X = torch.linspace(-5, 5, 100)
    g = torch.distributions.Normal(0, 1)
    Y = torch.exp(g.log_prob(X))
    ax.plot(X, Y, label="Unit Normal")
    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{post}}$")
    ax.set_ylabel("Count")
    ax.legend()
    addChi2(ax)
    save_func("post_global_pred_pulls_hist", fig)

    fig, ax = plt.subplots(layout="tight")
    h1 = np.histogram(p.numpy(), bins=10, range=(-5, 5), density=True)
    mplhep.histplot(h1, ax=ax, label="Global Pulls")
    if mask is not None:
        h2 = np.histogram(p[mask].numpy(), bins=10, range=(-5, 5), density=True)
        mplhep.histplot(h2, ax=ax, label="Blinded Pulls")
    ax.plot(X, Y, label="Unit Normal")
    ax.legend()
    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{post}}$")
    addChi2(ax)

    save_func("combo_pulls_hist", fig)

    global_chi2_pred = chi2Bins(bkg_mvn.mean, test_data.Y, summ["observed"]["std"])

    data = {"chi2_pred": float(global_chi2_pred)}
    return data


def chi2TestStat(post_pred, obs, **kwargs):
    # return post_pred.mean(dim=-1)
    return chi2Bins(post_pred, obs.Y, obs.V, min_var=1, power=1)


def makePValuePlots(trained_model, save_dir, test_stat=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    if test_stat is None:
        test_stat = chi2TestStat

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model = regression.loadModel(trained_model)
    all_data, train_mask = regression.getModelingData(trained_model)

    pred = regression.getPosteriorProcess(model, all_data, trained_model.transform)
    pred_samples = getPosteriorPred(pred, num_samples=500)
    post_pred = pred_samples["observed"]
    obs = all_data.Y
    dist = test_stat(post_pred, all_data).numpy()
    obs_stat = test_stat(all_data.Y, all_data).numpy()

    fig, ax = plt.subplots()

    density = gaussian_kde(dist)
    print(np.median(dist))
    print(obs_stat)
    xs = np.linspace(dist.min(), dist.max(), 200)

    density.covariance_factor = lambda: 0.25
    density._compute_covariance()

    ax.plot(xs, density(xs))
    fig.savefig(save_dir / "post_pred_density.png")

    data = {
        "chibins": {
            "median": float(np.median(dist)),
            "std": float(np.std(dist)),
            "obs": float(obs_stat),
        }
    }

    with open(save_dir / "post_pred_data.json", "w") as f:
        json.dump(data, f)


def runPValue(args):
    out = args.outdir or Path(args.input).parent
    m = torch.load(args.input)
    makePValuePlots(m, out)


def addPValueParser(parser):
    import argparse

    parser.add_argument(
        "-o", "--outdir", default=None, help="Output directory for plots"
    )
    parser.add_argument("input")
    parser.set_defaults(func=runPValue)
    return parser
