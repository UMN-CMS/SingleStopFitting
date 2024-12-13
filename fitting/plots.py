import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon

from .plot_tools import (
    addAxesToHist,
    getPolyFromSquares,
    makeSquares,
    plotData,
    plotRaw,
)


def plotPullDists(pred, raw_test, mask=None):
    pred_mean = pred.Y
    pred_variances = pred.V
    ret = {}
    fig, ax = plt.subplots(layout="tight")
    all_pulls = (raw_test.Y - pred_mean) / torch.sqrt(raw_test.V)
    p = all_pulls[torch.abs(all_pulls) < np.inf]
    p = all_pulls
    ax.hist(p, bins=np.linspace(-5.0, 5.0, 21), density=True)
    X = torch.linspace(-5, 5, 100)
    g = torch.distributions.Normal(0, 1)
    Y = torch.exp(g.log_prob(X))
    ax.plot(X, Y)

    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{o}}$")
    ax.set_ylabel("Count")
    ret["global_pulls_hist"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    window_p = all_pulls[mask]
    ax.hist(window_p, bins=np.linspace(-5.0, 5.0, 21), density=True)
    X = torch.linspace(-5, 5, 100)
    g = torch.distributions.Normal(0, 1)
    Y = torch.exp(g.log_prob(X))
    ax.plot(X, Y)

    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{o}}$")
    ax.set_ylabel("Count")
    ret["window_pulls_hist"] = (fig, ax)

    return ret


def makeDiagnosticPlots1D(pred, raw_test, raw_train, mask=None, inducing_points=None):
    ret = {}
    d = raw_test.dim

    def addWindow(ax):
        if mask is not None:
            w_max = raw_test.X[mask].max()
            w_min = raw_test.X[mask].min()

            m = raw_test.X > w_max
            if m.any():
                t = raw_test.X[m].min() - 10
                ax.axvline(t, 0, 1, ls="--", color="gray", alpha=0.5)

            m = raw_test.X < w_min
            if m.any():
                b = raw_test.X[m].max() + 10
                ax.axvline(b, 0, 1, ls="--", color="gray", alpha=0.5)

    def addInducing(ax):
        if inducing_points is not None:
            ax.scatter(inducing_points, np.zeros_like(inducing_points), c="red", s=1)

    pred_mean = pred.Y
    pred_variances = pred.V
    all_x2 = (pred_mean - raw_test.Y) ** 2 / raw_test.V
    x2 = torch.sum(all_x2)
    fig, ax = plt.subplots(layout="tight")
    addAxesToHist(ax, size=1.5)
    plotData(ax, raw_test, histtype="errorbar", color="black", label="Observed")
    ax.plot(pred.X, pred.Y, color="orange", label="GP Pred Mean")
    ax.fill_between(
        pred.X,
        pred.Y + torch.sqrt(pred.V),
        pred.Y - torch.sqrt(pred.V),
        color="orange",
        alpha=0.3,
        label="$\\pm\\sigma$",
    )
    addWindow(ax)
    ax.bottom_axes[0].set_ylim(-2, 2)
    ax.bottom_axes[0].plot(
        raw_test.X,
        (raw_test.Y - pred.Y) / torch.sqrt(raw_test.V),
        "o",
        color="black",
    )
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.bottom_axes[0].axhline(0, 0, 1, ls="--", color="gray", alpha=0.5)
    ax.bottom_axes[0].axhline(1, 0, 1, ls="-.", color="gray", alpha=0.5)
    ax.bottom_axes[0].axhline(-1, 0, 1, ls="-.", color="gray", alpha=0.5)
    addWindow(ax.bottom_axes[0])
    ax.legend()
    ret["summary_plot"] = (fig, ax)
    return ret


def makeDiagnosticPlots2D(pred, raw_test, raw_train, mask=None, inducing_points=None):
    ret = {}
    d = raw_test.dim
    if mask is not None and d == 2:
        squares = makeSquares(raw_test.X[mask], raw_test.E)
        points = getPolyFromSquares(squares)

    def addWindow(ax):
        if mask is None or d != 2:
            return
        else:
            poly = Polygon(points, edgecolor="green", linewidth=3, fill=False)
            ax.add_patch(poly)

    def addInducing(ax):
        if inducing_points is not None:
                ax.scatter(inducing_points[:, 0], inducing_points[:, 1], c="red", s=1)

    pred_mean = pred.Y
    pred_variances = pred.V
    all_x2 = (pred_mean - raw_test.Y) ** 2 / raw_test.V
    x2 = torch.sum(all_x2)

    fig, ax = plt.subplots(layout="tight")
    plotData(ax, raw_train)  # , norm=mpl.colors.LogNorm())
    ax.set_title("Masked Inputs (Training)")
    addWindow(ax)
    ret["training_points"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(ax, raw_test.E, raw_test.X, pred_mean)  # , norm=mpl.colors.LogNorm())
    ax.set_title("GPR Mean Prediction")
    addWindow(ax)
    ret["gpr_mean"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(
        ax,
        raw_test.E,
        raw_test.X,
        pred_mean,
    )  # norm=mpl.colors.LogNorm())
    ax.set_title("GPR Mean Prediction")
    addInducing(ax)
    ret["inducing_gpr_mean"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    plotRaw(
        ax,
        raw_test.E,
        raw_test.X,
        raw_test.Y,
    )  # norm=mpl.colors.LogNorm())
    ax.set_title("Observed Outputs")
    addWindow(ax)
    ret["observed_outputs"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(ax, raw_test.E, raw_test.X, raw_test.V)
    ax.set_title("Observed Variances")
    addWindow(ax)
    ret["observed_variances"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(ax, raw_test.E, raw_test.X, pred.V)
    ax.set_title("Pred Variances")
    addWindow(ax)
    ret["predicted_variances"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(
        ax, raw_test.E, raw_test.X, torch.sqrt(pred.V) / raw_test.Y, cmin=0, cmax=0.1
    )
    ax.set_title("Relative Uncertainty (std/val)")
    addWindow(ax)
    ret["relative_uncertainty"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = plotRaw(
        ax, raw_test.E, raw_test.X, torch.sqrt(raw_test.V) / raw_test.Y
    )
    ax.set_title("Relative Stat Uncertainty (std/val)")
    addWindow(ax)
    ret["relative_stat_uncertainty"] = (fig, ax)

    ret.update(plotPullDists(pred, raw_test, mask=mask))
    return ret


def makeDiagnosticPlots(pred, raw_test, raw_train, mask=None, inducing_points=None):
    d = raw_test.dim
    if d == 1:
        return makeDiagnosticPlots1D(
            pred, raw_test, raw_train, mask=mask, inducing_points=inducing_points
        )
    elif d == 2:
        return makeDiagnosticPlots2D(
            pred, raw_test, raw_train, mask=mask, inducing_points=inducing_points
        )


def makeNNPlots(model, test_data):
    ret = {}
    fig, ax = plt.subplots(layout="tight")
    fe = model.covar_module.base_kernel.base_kernel.feature_extractor
    T = fe(test_data.X).detach()
    fig, ax = plt.subplots()
    ax.scatter(T[:, 0], T[:, 1], c=test_data.Y, cmap="hsv")
    ret["NN"] = (fig, ax)
    return ret
