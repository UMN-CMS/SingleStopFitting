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


def extractSigs(d, points, coupling="312", injection=1.0):
    ret = {
        (signal_id.mt, signal_id.mx): next(
            (
                x
                for x in d[signal_id]["injections"]
                if x["signal_injected"] == injection and signal_id.coupling == coupling
            ),
            {"sig": None},
        ).get("sig")
        for signal_id in points
    }
    return ret


def extractSigInjected(d, points, coupling="312", injection=1):
    ret = {
        (signal_id.mt, signal_id.mx): next(
            (
                (
                    x
                    for x in d[signal_id]["injections"]
                    if x["signal_injected"] == 0.0 and signal_id.coupling == coupling
                )
            ),
            None,
        )
        for signal_id in points
    }
    ret = {
        k: next(x[1] for x in v["sig_inject"] if x[0] == injection)
        for k, v in ret.items()
        if v and "sig_inject" in v
    }

    return ret


def main():
    mplhep.style.use("CMS")
    # args = parseAguments()
    # with open(args.input) as f:
    #     data = json.load(f)
    # plotRate(data, args.output, coupling=args.coupling)
    with open("sr_gathered.json", "r") as f:
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
    injected_sigs_uncomp = extractSigInjected(data, points_uncomp, injection=1)
    sigs_uncomp = extractSigs(data, points_uncomp, injection=1)
    points_comp = [d for d in data if d.algo == "comp" and d.mx != 100]
    injected_sigs_comp = extractSigInjected(data, points_comp, injection=1)
    sigs_comp = extractSigs(data, points_comp, injection=1)

    points_pref = [
        d
        for d in data
        if (
            (d.algo == "uncomp" and d.mx / d.mt < 0.75)
            or (d.algo == "comp" and d.mx / d.mt > 0.75)
        )
        and d.mx > 300
    ]
    injected_sigs_pref = extractSigInjected(data, points_pref, injection=1)
    sigs_pref = extractSigs(data, points_pref, injection=1)
    print(sigs_pref)

    plotSig(injected_sigs_uncomp, "deletemelater/asimov_312sig_uncomp.png")
    plotSig(sigs_uncomp, "deletemelater/srmc_312sig_uncomp.png")

    plotSig(injected_sigs_comp, "deletemelater/asimov_312sig_comp.png")
    plotSig(sigs_comp, "deletemelater/srmc_312sig_comp.png")

    plotSig(injected_sigs_pref, "deletemelater/asimov_312sig_pref.png")
    plotSig(sigs_pref, "deletemelater/srmc_312sig_pref.png")

    # data = {
    #     x: (
    #         y / sigs_comp[x]
    #         if (
    #             x in sigs_comp
    #             and y is not None
    #             and sigs_comp[x] is not None
    #             and sigs_comp[x] > 0
    #         )
    #         else None
    #     )
    #     for x, y in sigs_uncomp.items()
    # }
    # plotSigRatio(data, "deletemelater/asimov_312sig_ratio.png")


if __name__ == "__main__":
    main()
