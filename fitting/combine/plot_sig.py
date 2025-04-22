import json

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib
import mplhep
import numpy as np
from rich import print
from scipy.interpolate import griddata

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
    data = np.array([[*x, y] for x, y in data.items()])
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=400)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Significance")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, loc=1)

    fig.savefig(output_path)


def extractSigs(d, points, coupling="312", injection=1.0):
    ret = {
        (signal_id.mt, signal_id.mx): next(
            (x for x in d[signal_id]["injections"] if 
            x["signal_injected"] == injection
            and signal_id.coupling == coupling)
            , {"sig":None}).get("sig")
        for signal_id in points
    }
    return ret


def plotSigRatio(data, output_path, coupling="312"):
    data = np.array([[*x, y] for x, y in data.items()])
    fig, ax = plt.subplots()
    c = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=800, vmin=0, vmax=2)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Significance Ratio")
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    addCMS(ax, loc=1)

    for x,y,z in data:
        if z is not None:
            ax.text(x,y, f"{z:0.2f}", ha="center",size=8)

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
    with open("gathered.json", "r") as f:
        data = json.load(f)
    data = {SignalId(**x["signal_info"]): x for x in data}
    points = [
        SignalId("uncomp", "312", 1200, 700),
        SignalId("uncomp", "312", 1000, 400),
        SignalId("uncomp", "312", 1500, 600),
        # SignalId("uncomp", "312", 2000, 400),
        SignalId("uncomp", "312", 2000, 1200),
        SignalId("comp", "312", 1200, 1100),
        SignalId("comp", "312", 1500, 1450),
        # SignalId("comp", "312", 2000, 1900),
    ]
    points_uncomp = [d for d in data if d.algo == "uncomp" and d.mx != 100]
    sigs_uncomp = extractSigs(data, points_uncomp, injection=1.0)

    plotSig(sigs_uncomp, "deletemelater/312sig_uncomp.png")
    points_comp = [d for d in data if d.algo == "comp" and d.mx != 100]

    sigs_comp = extractSigs(data, points_comp, injection=1.0)
    plotSig(sigs_comp, "deletemelater/312sig_comp.png")

    data = {
        x: (
            y / sigs_comp[x]
            if (
                x in sigs_comp
                and y is not None
                and sigs_comp[x] is not None
                and sigs_comp[x] > 0
            )
            else None
        )
        for x, y in sigs_uncomp.items()
    }
    plotSigRatio(data, "deletemelater/312sig_ratio.png")
    # points = [
    #     SignalId("uncomp", "313", 1000, 400),
    #     SignalId("uncomp", "313", 1500, 600),
    #     SignalId("uncomp", "313", 2000, 600),
    #     SignalId("comp", "313", 2000, 1900),
    #     SignalId("comp", "313", 1500, 1400),
    #     # SignalId("comp", "312", 2000, 1900),
    # ]
    # plotRates(data, points, "deletemelater/313rates.png")


if __name__ == "__main__":
    main()
