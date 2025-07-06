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
from scipy.interpolate import griddata

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


# def interpolate(
#     vals,
#     method="nearest",
#     boundary=(
#         (start_points_x, start_points_y),
#         (end_x_axis, start_points_y),
#         (end_x_axis, end_x_axis),
#         (start_points_x, start_points_x),
#     ),
#     x_step=5,
#     y_step=5,
# ):
#     out = [x for for x in range(boundary[0][0], boundary[2][0])]
#     d = griddata(vals[:, :2], vals[:, 2], (grid_x, grid_y), method="nearest")


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


def plotSig(data, output_path, coupling="312", year="2018"):
    data = np.array(
        [
            [
                x.signal_point.mt,
                x.signal_point.mx,
                x.inference_data.get("significance", np.nan),
            ]
            for x in data
        ]
    )
    print(data)
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=400)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Significance")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, year=year, loc=1)
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
                x.inference_data.get("limit", {"50": np.nan})["50"],
            ]
            for x in data
        ]
    )
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=400)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Expected Limit")
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


def main():
    mplhep.style.use("CMS")
    # args = parseAguments()
    # with open(args.input) as f:
    #     data = json.load(f)
    # plotRate(data, args.output, coupling=args.coupling)
    with open("gathered/2025_07_05_small_nn_official.json", "r") as f:
        base_data = SignalRunCollection.model_validate_json(f.read())
    dropped_points = [(1000, 800), (1200, 800)]
    years = [
        # "2016_preVFP",
        "2016_postVFP",
        "2017",
        "2018",
        "2022_preEE",
        "2022_postEE",
        "2023_preBPix",
        "2023_postBPix",
    ]

    for coupling, year in it.product(["312", "313"], years):

        def f(item):
            return (
                item.signal_injected == 1
                and item.signal_point.coupling == coupling
                and item.signal_point.mx > 300
                and item.signal_point.mt > 900
                and (item.signal_point.mt, item.signal_point.mx) not in dropped_points
            )

        data = base_data.filter(year=year, other_filter=f)
        Path(f"deletemelater/{year}/").mkdir(exist_ok=True, parents=True)
        plotSig(
            data,
            f"deletemelater/{year}/srmc_sig_{coupling}_plot.pdf",
            coupling=coupling,
            year=year,
        )

        def f(item):
            return (
                item.signal_injected == 0
                and item.signal_point.coupling == coupling
                and item.signal_point.mx > 300
                and item.signal_point.mt > 900
                and (item.signal_point.mt, item.signal_point.mx) not in dropped_points
            )

        data_0 = base_data.filter(year=year, other_filter=f)
        plotLim(
            data_0,
            f"deletemelater/{year}/srmc_lim_{coupling}_plot.pdf",
            coupling=coupling,
            year=year,
        )


if __name__ == "__main__":
    main()
