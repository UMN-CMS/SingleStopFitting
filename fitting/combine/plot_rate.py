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


def plotRates(data, signal_ids, output_path, coupling="312"):
    data = {
        signal_id: np.array(
            sorted(
                [
                    (x["signal_injected"], x["fit"]["r"])
                    for x in data[signal_id]["injections"]
                ]
            )
        )
        for signal_id in signal_ids
    }
    print(data)

    mplhep.style.use("CMS")

    # output_path = Path(output_path)
    # output_path.parent.mkdir(exist_ok=True, parents=True)
    #
    # xyz = [
    #     [x["signal"][1], x["signal"][2], tryGet(x["props"]["r"], 0)]
    #     for x in data
    #     if x["signal"][0] == coupling
    # ]
    #
    # points = [[x, y] for x, y, z in xyz]
    # values = [z for x, y, z in xyz]
    # print(points)
    #
    # grid_x, grid_y = np.mgrid[1000:2000:25, 100:2000:25]
    # mask = grid_x > grid_y
    # print(mask.shape)
    # # grid_x,grid_y = grid_x[mask], grid_y[mask]
    # # grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    # # # grid_x,grid_y,grid_z= grid_x[mask], grid_y[mask], grid_z[mask]
    # # grid_z[~mask]  = np.nan
    # fig, ax = plt.subplots()
    # ax.set_xlabel(r"$m_{\tilde{t}}$")
    # ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    # pm = ax.pcolormesh(grid_x, grid_y, grid_z, shading="nearest")
    # c = ax.scatter(
    #     [x[0] for x in points],
    #     [x[1] for x in points],
    #     c=values,
    #     s=400,
    #     # norm=matplotlib.colors.LogNorm(),
    # )
    # cb = fig.colorbar(c, ax=ax)
    # cb.set_label(r"Psuedodata Observed Rate")

    fig, ax = plt.subplots()
    for title, points in data.items():
        ax.plot(points[:, 0], points[:, 1], label=formatSignal(title))
    ax.legend()
    ax.set_xlabel("Signal Injected (Relative to $\lambda_{31j}''=0.1$)")
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
    # data[SignalId("uncomp", "312", "1200", "400")]
    plotRates(data, points, "deletemelater/312rates.png")
    points = [
        SignalId("uncomp", "313", 1000, 400),
        SignalId("uncomp", "313", 1500, 600),
        SignalId("uncomp", "313", 2000, 600),
        SignalId("comp", "313", 2000, 1900),
        SignalId("comp", "313", 1500, 1400),
        # SignalId("comp", "312", 2000, 1900),
    ]
    plotRates(data, points, "deletemelater/313rates.png")


if __name__ == "__main__":
    main()
