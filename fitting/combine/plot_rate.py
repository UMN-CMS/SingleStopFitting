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
    return f"$\lambda''_{{{sid.coupling}}}-{sid.mt},{sid.mx}$ {sid.algo}"


def plotRates(data, signal_ids, output_path, coupling="312"):
    data = {
        signal_id: np.array(
            sorted(
                [
                    (x["signal_injected"], (x.get("fit") or {"r": None})["r"])
                    for x in data[signal_id]["injections"]
                ]
            )
        )
        for signal_id in signal_ids
    }
    mplhep.style.use("CMS")

    fig, ax = plt.subplots()
    for title, points in data.items():
        print(points)
        points = points[(points[:, 1]) != np.array(None)]
        ax.plot(points[:, 0], points[:, 1], "o-", label=formatSignal(title))
    ax.plot([0, 16], [0, 16], "--", color="gray", label="y=x")
    ax.legend()
    ax.set_xlabel(f"Signal Injected (Relative to $\lambda_{{{coupling}}}''=0.1$)")
    ax.set_ylabel("Signal Extracted")
    addCMS(ax)

    fig.savefig(output_path)


def plotInjectedRates(data, signal_ids, output_path, coupling="312"):
    data = {
        signal_id: np.array(
            sorted(
                next(
                    (
                        x["fit_inject"]
                        for x in data[signal_id]["injections"]
                        if x["signal_injected"] == 0.0 and "fit_inject" in x
                    ),
                    [],
                )
            )
        )
        for signal_id in signal_ids
    }
    mplhep.style.use("CMS")

    fig, ax = plt.subplots()
    for title, points in data.items():
        if len(points):
            ax.plot(points[:, 0], points[:, 1], label=formatSignal(title))
    ax.legend()
    ax.set_xlabel(f"Signal Injected (Relative to $\lambda_{{{coupling}}}''=0.1$)")
    ax.set_ylabel("Signal Extracted")
    addCMS(ax)

    fig.savefig(output_path)


def parseAguments():
    parser = argparse.ArgumentParser(description="make plot")
    parser.add_argument("-o", "--output", required=True, type=str, help="")
    parser.add_argument("-c", "--coupling", type=str, help="", default="312")
    parser.add_argument("input")

    return parser.parse_args()


def main():
    # args = parseAguments()
    # with open(args.input) as f:
    #     data = json.load(f)

    # plotRate(data, args.output, coupling=args.coupling)
    with open("gathered/srmc_gathered.json", "r") as f:
        data = json.load(f)
    data = {SignalId(**x["signal_info"]): x for x in data}
    points = [
        # SignalId("uncomp", "312", 1200, 700),
        SignalId("uncomp", "312", 1000, 400),
        SignalId("uncomp", "312", 1200, 400),
        SignalId("uncomp", "312", 1300, 600),
        SignalId("uncomp", "312", 1400, 400),
        SignalId("uncomp", "312", 1500, 400),
        SignalId("uncomp", "312", 1500, 600),
        SignalId("uncomp", "312", 2000, 400),
        SignalId("uncomp", "312", 2000, 1200),
        SignalId("comp", "312", 1300, 1200),
        SignalId("comp", "312", 1500, 1400),
        SignalId("comp", "312", 1400, 1300),
        SignalId("comp", "312", 2000, 1900),
    ]
    # data[SignalId("uncomp", "312", "1200", "400")]
    plotInjectedRates(data, points, "deletemelater/rates_asimov_312.png", coupling=312)
    plotRates(data, points, "deletemelater/rates_srmc_312.png", coupling=312)
    points = [
        SignalId("uncomp", "313", 1000, 400),
        # SignalId("uncomp", "313", 1500, 600),
        SignalId("uncomp", "313", 2000, 600),
        SignalId("comp", "313", 2000, 1900),
        SignalId("comp", "313", 1500, 1400),
        # SignalId("comp", "312", 2000, 1900),
    ]
    plotInjectedRates(data, points, "deletemelater/rates_asimov_313.png", coupling=313)
    plotRates(data, points, "deletemelater/rates_srmc_313.png", coupling=313)


if __name__ == "__main__":
    main()
