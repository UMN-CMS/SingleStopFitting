import json
from rich import print
import argparse
from collections import defaultdict

from fitting.plotting.annots import addCMS
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib
import mplhep
import numpy as np
from scipy.interpolate import griddata
from collections import namedtuple
from fitting.core import SignalRunCollection, SignalPoint

SignalId = namedtuple("SignalId", "algo coupling mt mx")


def formatSignalAndSpread(sid, spread):
    return f"$\lambda''_{{{sid.coupling}}}-{sid.mt},{sid.mx} (\\sigma={spread})$ "


def formatSignalAndInject(sid, r):
    return f"$\lambda''_{{{sid.coupling}}}-{sid.mt},{sid.mx} (r={r})$ "


def plotRates(data, output_path, coupling="312"):
    mplhep.style.use("CMS")

    stats = {k: (np.mean(v), np.std(v)) for k, v in data.items()}
    new = defaultdict(list)
    for k, v in stats.items():
        new[(k[0], k[2])].append([k[1], *v])
    for k in new:
        new[k].sort(key=lambda x: x[0])
        new[k] = np.array(new[k])

    fig, ax = plt.subplots()
    for k, v in new.items():
        ax.errorbar(
            v[:, 0], v[:, 1], yerr=v[:, 2], fmt="o", label=formatSignalAndSpread(*k)
        )

    ax.plot(
        [0, 16],
        [0, 16],
        "--",
        color="gray",
        label="Extracted = Injected",
    )
    ax.legend()
    ax.set_xlabel(f"Signal Injected (Relative to $\lambda_{{{coupling}}}''=0.1$)")
    ax.set_ylabel("Signal Extracted")
    addCMS(ax)
    fig.savefig(output_path)
    plt.close(fig)


def plotSigs(data, output_path, coupling="312"):
    mplhep.style.use("CMS")

    stats = {k: (np.mean(v), np.std(v)) for k, v in data.items()}
    new = defaultdict(list)
    for k, v in stats.items():
        new[(k[0], k[1])].append([k[2], *v])
    for k in new:
        new[k].sort(key=lambda x: x[0])
        new[k] = np.array(new[k])

    fig, ax = plt.subplots()
    for k, v in new.items():
        ax.errorbar(
            v[:, 0], v[:, 1], yerr=v[:, 2], fmt="o", label=formatSignalAndInject(*k)
        )

    ax.legend()
    ax.set_xlabel(f"Window Size")
    ax.set_ylabel("Significance")
    ax.set_ylim(bottom=0)
    addCMS(ax)
    fig.savefig(output_path)
    plt.close(fig)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-r", "--rate-injected", type=float)
    parser.add_argument("-o", "--output")
    return parser.parse_args()


def getRate(collection):
    return np.array(
        [x.inference_data.get("extracted_rate", {}).get("r") for x in collection]
    )


def getSig(collection):
    d = [x.inference_data.get("significance") for x in collection]
    return np.array([x for x in d if x is not None])


def main():
    args = parseArgs()
    with open(args.input, "r") as f:
        base_data = SignalRunCollection.model_validate_json(f.read())

    for p in [(1200, 600), (1500, 600), (2000, 1900), (1400, 1300)]:
        for s in [1.0, 1.5, 1.75, 2.0, 2.5]:
            data = base_data.filter(
                signal_id=SignalPoint(coupling="312", mt=p[0], mx=p[1]), spread=s
            )
            grouped = data.groupby(
                lambda x: (x.signal_point, x.signal_injected, x.metadata.window.spread)
            )
            rates = {k: getRate(v) for k, v in grouped.items()}
            plotRates(
                rates,
                f"deletemelater/rates_srmc_312_{p[0]}_{p[1]}_{str(round(s,2)).replace('.','p')}.png",
                coupling=312,
            )
        for s in [1.0, 9.0, 16.0]:
            data = base_data.filter(
                signal_id=SignalPoint(coupling="312", mt=p[0], mx=p[1]),
                signal_injected=s,
                # other_filter=lambda x: x.signal_injected > 0,
            )
            grouped = data.groupby(
                lambda x: (x.signal_point, x.signal_injected, x.metadata.window.spread)
            )
            rates = {k: getSig(v) for k, v in grouped.items()}
            plotSigs(
                rates,
                f"deletemelater/sig_srmc_312_{p[0]}_{p[1]}_{str(round(s,2)).replace('.','p')}.png",
                coupling=312,
            )


if __name__ == "__main__":
    main()
