import json
import itertools as it

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib
import mplhep
import numpy as np
from rich import print
from scipy.interpolate import griddata, RBFInterpolator

from fitting.core import SignalRunCollection

import json

from fitting.plotting.annots import addCMS
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib
import mplhep
import numpy as np
from scipy.interpolate import griddata
from collections import namedtuple

SignalId = namedtuple("SignalId", "algo coupling mt mx")


def formatSignal(sid):
    return f"$\lambda''_{{{sid.coupling}}}({sid.mt},{sid.mx})$ {sid.algo}"


start_points_x = 950
start_points_y = 350
start_xaxis = 500
end_xaxis = 2100
start_yaxis = 100
end_yaxis = 2000


def rbfInterpolate(
    vals,
    method="nearest",
    boundary=(
        (start_points_x, start_points_y),
        (end_xaxis, start_points_y),
        (end_xaxis, end_xaxis),
        (start_points_x, start_points_x),
    ),
    x_step=20,
    y_step=20,
):

    x = np.arange(boundary[0][0], boundary[2][0], x_step)
    y = np.arange(boundary[0][1], boundary[1][0], y_step)
    X, Y = np.meshgrid(x, y)
    m_X, m_Y = np.meshgrid(np.diff(x) / 2 + x[:-1], np.diff(y) / 2 + y[:-1])
    shape = m_X.shape
    interpolation_points = np.stack([m_X, m_Y], axis=2)
    interpolation_points = interpolation_points.reshape(
        -1, interpolation_points.shape[-1]
    )
    # good = Y < X
    # interpolation_points = np.array(
    #     [
    #         (x, y)
    #         for x in range(boundary[0][0], boundary[2][0], x_step)
    #         for y in range(start_points_y, x, y_step)
    #     ]
    # )
    interpolator = RBFInterpolator(
        vals[:, :2], vals[:, 2], kernel="cubic", smoothing=5.0
    )
    interpolated = interpolator(interpolation_points)
    reshaped = interpolated.reshape(shape)
    return X, Y, np.ma.masked_where(m_Y > m_X, reshaped)

    # return np.concatenate([interpolation_points, interpolated[:, np.newaxis]], axis=1)


def commonElements(ax):
    ax.set_xlim(start_xaxis, end_xaxis)
    ax.set_ylim(start_yaxis, end_yaxis)

    rect_dijet = patches.Rectangle(
        (start_points_x, start_yaxis),
        (end_xaxis - start_xaxis),
        (start_points_y - start_yaxis),
        edgecolor="none",
        facecolor="gray",
        alpha=0.3,
    )
    ax.add_patch(rect_dijet)

    rect_scout = patches.Polygon(
        [
            (start_xaxis, start_yaxis),
            (start_points_x, start_yaxis),
            (start_points_x, start_points_x),
            (start_xaxis, start_xaxis),
        ],
        edgecolor="none",
        facecolor="gray",
        alpha=0.2,
    )
    ax.add_patch(rect_scout)

    ax.text(
        start_points_x + (end_xaxis - start_points_x) / 2,
        start_yaxis + (start_points_y - start_yaxis) / 2,
        "Dijet Regime",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.annotate(
        "",
        (
            start_points_x + 0.8 * (end_xaxis - start_points_x),
            start_yaxis + 0.15 * (start_points_y - start_yaxis),
        ),
        (
            start_points_x + 0.8 * (end_xaxis - start_points_x),
            start_yaxis + 0.85 * (start_points_y - start_yaxis),
        ),
        arrowprops=dict(arrowstyle="->", lw=2),
    )
    ax.annotate(
        "",
        (
            start_points_x + 0.2 * (end_xaxis - start_points_x),
            start_yaxis + 0.15 * (start_points_y - start_yaxis),
        ),
        (
            start_points_x + 0.2 * (end_xaxis - start_points_x),
            start_yaxis + 0.85 * (start_points_y - start_yaxis),
        ),
        arrowprops=dict(arrowstyle="->", lw=2),
    )

    ax.text(
        start_xaxis + (start_points_x - start_xaxis) / 2,
        start_yaxis + (start_points_x - start_yaxis) / 3,
        "Scouting\n Regime",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.annotate(
        "",
        (
            start_xaxis + 0.15 * (start_points_x - start_xaxis),
            start_yaxis + 0.07 * (start_points_x - start_yaxis),
        ),
        (
            start_xaxis + 0.65 * (start_points_x - start_xaxis),
            start_yaxis + 0.2 * (start_points_x - start_yaxis),
        ),
        arrowprops=dict(arrowstyle="->", lw=2),
    )


def plotSig(
    data,
    output_path,
    coupling="312",
    year="2018",
    drop_if_greater=None,
    interpolate=False,
):
    data = np.array(
        [
            [
                x.signal_point.mt,
                x.signal_point.mx,
                x.inference_data.get("significance", np.nan),
            ]
            for x in data
            if (
                not drop_if_greater
                or x.inference_data.get("significance", np.nan) < drop_if_greater
            )
        ]
    )
    fig, ax = plt.subplots()
    if interpolate:
        X, Y, C = rbfInterpolate(data)
        c = ax.pcolormesh(
            X,
            Y,
            C,
            # cmap="managua",
            # cmin=0,
            # cmax=10,
        )
    else:
        c = ax.scatter(
            data[:, 0],
            data[:, 1],
            c=data[:, 2],
            s=400,
            # cmap="managua",
            # vmin=0,
            # vmax=10,
        )

    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Significance")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, year=year, loc=1, coupling=coupling)
    commonElements(ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plotLim(data, output_path, coupling="312", year="2018"):
    data = np.array(
        [
            [
                x.signal_point.mt,
                x.signal_point.mx,
                0.1 * x.inference_data.get("limit", {"50": np.nan})["50"] ** 2,
            ]
            for x in data
        ]
    )
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=400)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(f"95% CLs Upper Limit on $\lambda_{{{coupling}}}''$")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, loc=1, year=year)

    commonElements(ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parseAguments():
    parser = argparse.ArgumentParser(description="make plot")
    parser.add_argument("-o", "--output", required=True, type=str, help="")
    parser.add_argument("-c", "--coupling", type=str, help="", default="312")
    parser.add_argument("input")

    return parser.parse_args()


def main(args):
    infile = args.input
    outdir = Path(args.outdir)
    mplhep.style.use("CMS")

    with open(infile, "r") as f:
        base_data = SignalRunCollection.model_validate_json(f.read())

    years = [
        "2016_preVFP",
        "2016_postVFP",
        "2017",
        "2018",
        "2022_preEE",
        "2022_postEE",
        "2023_preBPix",
        "2023_postBPix",
    ]

    dropped_points = [(1000, 800), (1200, 800)]

    for coupling, year in it.product(["312", "313"], years):
        year_outdir = outdir / str(year)
        year_outdir.mkdir(exist_ok=True, parents=True)

        def makeFilter(r):
            def f(item):
                return (
                    item.signal_injected == r
                    and item.signal_point.coupling == coupling
                    and item.signal_point.mx > 300
                    and item.signal_point.mt > 900
                    and (item.signal_point.mt, item.signal_point.mx)
                    not in dropped_points
                )

            return f

        plotSig(
            base_data.filter(year=year, other_filter=makeFilter(1)),
            year_outdir / f"srmc_sig_r_1p0_{coupling}_plot.pdf",
            coupling=coupling,
            year=year,
            drop_if_greater=10,
            interpolate=False,
        )
        plotSig(
            base_data.filter(year=year, other_filter=makeFilter(16)),
            year_outdir / f"srmc_sig_r_16p0_{coupling}_plot.pdf",
            coupling=coupling,
            year=year,
        )
        plotLim(
            base_data.filter(year=year, other_filter=makeFilter(0)),
            year_outdir / f"srmc_lim_{coupling}_plot.pdf",
            coupling=coupling,
            year=year,
        )


def addPlotSensitivityParser(parser):
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("input")
    parser.set_defaults(func=main)


if __name__ == "__main__":
    main()
