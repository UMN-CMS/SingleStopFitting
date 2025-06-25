import json

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib
import mplhep
import numpy as np
from rich import print
from scipy.interpolate import griddata

from fitting.core import signal_run_list_adapter

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


def plotSig(data, output_path, coupling="312"):
    data = np.array(
        [
            [x.signal_point.mt, x.signal_point.mx, x.inference_data["significance"]]
            for x in data
        ]
    )
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=400)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Significance")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, loc=1)

    fig.savefig(output_path)


def plotLim(data, output_path, coupling="312"):
    data = np.array(
        [
            [x.signal_point.mt, x.signal_point.mx, x.inference_data["limit"]]
            for x in data
        ]
    )
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=400)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Expected Limit")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, loc=1)

    fig.savefig(output_path)


def plotSigRatio(data, output_path, coupling="312"):
    data = np.array([[*x, y] for x, y in data.items()])
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=800, vmin=0, vmax=2)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Significance Ratio")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, loc=1)

    for x, y, z in data:
        if z is not None:
            ax.text(x, y, f"{z:0.2f}", ha="center", size=8)

    fig.savefig(output_path)


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
    with open("gathered/2025_06_19_srmc_small_nn.json", "r") as f:
        data = signal_run_list_adapter.validate_json(f.read())

    dropped_points = [(1000, 800)]

    def filter(item):
        return (
            item.signal_injected == 1
            and item.signal_point.coupling == "312"
            and item.signal_point.mx > 300
            and (item.signal_point.mt, item.signal_point.mx) not in dropped_points
        )

    data_4 = sorted([x for x in data if filter(x)], key=lambda x: x.signal_point)

    plotSig(data_4, "deletemelater/srmc_312_sig_plot.png")

    def filter(item):
        return (
            item.signal_injected == 0
            and item.signal_point.coupling == "312"
            and item.signal_point.mx > 300
            and (item.signal_point.mt, item.signal_point.mx) not in dropped_points
        )


    data_0 = sorted([x for x in data if filter(x)], key=lambda x: x.signal_point)
    print(data_0)
    plotLim(data_0, "deletemelater/srmc_312_lim_plot.png")


if __name__ == "__main__":
    main()
