import pickle as pkl
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import regression
from .plot_tools import plotData
from .windowing import EllipseWindow
from matplotlib.patches import Polygon

from .plot_tools import (
    addAxesToHist,
    getPolyFromSquares,
    makeSquares,
    plotData,
    plotRaw,
)


Window = namedtuple("Window", "center axes")


def makeWindow1D(signal_data, frac=0.68):
    max_idx = torch.argmax(signal_data.Y)
    max_x = signal_data.X[max_idx]
    print(max_x)

    rough_window = torch.tensor([max_x - 400.0, max_x + 400.0])
    mask = (signal_data.X > rough_window[0]) & (signal_data.X < rough_window[1])
    masked_x, masked_y = signal_data.X[mask], signal_data.Y[mask]
    total = torch.sum(signal_data.Y)
    print(total)
    curr_best = (0, 999999)
    for x in masked_x:
        greater = masked_x > x
        for y in masked_x[greater]:
            total_mask = (masked_x > x) & (masked_x < y)
            window_total = torch.sum(masked_y[total_mask])
            win_frac = window_total / total

            if win_frac > frac and (y - x) < (curr_best[1] - curr_best[0]):
                print(f"{curr_best}, {win_frac}")
                curr_best = (x, y)
    diff = curr_best[1] - curr_best[0]
    return EllipseWindow(curr_best[0] + diff / 2, diff / 2)


def windowPlot1D(signal_data, window, frac=None):
    fig, ax = plt.subplots()
    plotData(ax, signal_data)
    if bounds[0]:
        ax.axvline(window.center - window.axes, 0, 1, color="black")
    if bounds[1]:
        ax.axvline(window.center + window.axes, 0, 1, color="black")
    return fig, ax


def windowPlot2D(signal_data, window, frac=None):
    fig, ax = plt.subplots()
    plotData(ax, signal_data)
    mask = window(*torch.unbind(signal_data.X, dim=-1))
    squares = makeSquares(signal_data.X[mask], signal_data.E)
    points = getPolyFromSquares(squares)
    poly = Polygon(points, edgecolor="green", linewidth=3, fill=False)
    ax.add_patch(poly)

    return fig, ax


def makeWindow2D(signal_data, frac=0.4):
    total = torch.sum(signal_data.Y)
    peak_x = signal_data.X[signal_data.Y.argmax()]
    min_x = torch.transpose(signal_data.X, 0, 1).min(dim=1).values
    max_x = torch.transpose(signal_data.X, 0, 1).max(dim=1).values
    diff = max_x - min_x
    w = 0.15 * diff
    ll, ur = peak_x - w, peak_x + w

    mask = (signal_data.X > ll).all(dim=1) & (signal_data.X < ur).all(dim=1)
    masked_x, masked_y = signal_data.X[mask], signal_data.Y[mask]

    step_x0 = (masked_x[:, 0][masked_x[:, 0] > peak_x[0]] - peak_x[0])[::4]
    step_x1 = (masked_x[:, 1][masked_x[:, 1] > peak_x[1]] - peak_x[1])[::4]

    curr_best = (0, 999999)
    curr_best_size = 999999999
    best_window = None
    print(step_x0.shape)
    print(torch.cartesian_prod(step_x0, step_x1).shape)
    
    for a, b in torch.cartesian_prod(step_x0, step_x1):
        a = a-0.01
        b = b-0.01
        w = EllipseWindow(peak_x, torch.tensor([a, b]))
        m = w(*torch.unbind(masked_x, dim=-1))
        window_total = masked_y[m].sum()
        win_frac = window_total / total
        size = torch.count_nonzero(m)
        # print(win_frac, w)
        if win_frac > frac and size < curr_best_size:
            print(f"{curr_best}, {win_frac}, {size}")
            print(w)
            curr_best = (a, b)
            curr_best_size = size
            best_window=w
    print("Done")
    return best_window


def main():
    with open(
        "regression_results/2018_Signal312_m14_m.pkl",
        "rb",
    ) as f:
        signal312 = pkl.load(f)

    with open(
        "regression_results/2018_Signal312_nn_uncomp_0p67_m14_vs_mChiUncompRatio.pkl",
        "rb",
    ) as f:
        signal312_2d = pkl.load(f)

    # signal_hist_names = [
    #     "signal_312_1200_1100",
    #     "signal_312_1500_1400",
    #     "signal_312_2000_1900",
    # ]
    signal_hist_names = [
        "signal_312_1200_400",
        "signal_312_1200_800",
        "signal_312_1500_400",
        "signal_312_1500_600",
        "signal_312_1500_1000",
        "signal_312_2000_900",
        "signal_312_2000_1200",
    ]
    signals_to_scan = [
        (
            sn,
            signal312_2d[sn, "Signal312"]["hist_collection"]["histogram"][
                "central", ...
            ],
        )
        for sn in signal_hist_names
    ]
    output_dir = Path("signal_plots")
    output_dir.mkdir(exist_ok=True, parents=True)

    for n, h in signals_to_scan:
        signal_regression_data, *_ = regression.makeRegressionData(h)
        window = makeWindow2D(signal_regression_data)
        # window = makeWindow1D(signal_regression_data)
        # fig,ax = windowPlot1D(signal_regression_data, window)
        fig, ax = windowPlot2D(signal_regression_data, window)
        fig.savefig(output_dir/f"{n}.pdf")


if __name__ == "__main__":
    main()
