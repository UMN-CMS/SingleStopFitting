import json

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib
import mplhep
import numpy as np
from scipy.interpolate import griddata


def tryGet(obj, idx):
    try:
        return obj[idx]
    except IndexError as e:
        return None


def plotRate(data, output_path, coupling="312"):
    mplhep.style.use("CMS")

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    xyz = [
        [x["signal"][1], x["signal"][2], tryGet(x["props"]["r"], 0)]
        for x in data
        if x["signal"][0] == coupling
    ]

    points = [[x, y] for x, y, z in xyz]
    values = [z for x, y, z in xyz]
    print(points)

    grid_x, grid_y = np.mgrid[1000:2000:25, 100:2000:25]
    mask = grid_x > grid_y
    print(mask.shape)
    # grid_x,grid_y = grid_x[mask], grid_y[mask]
    # grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    # # grid_x,grid_y,grid_z= grid_x[mask], grid_y[mask], grid_z[mask]
    # grid_z[~mask]  = np.nan
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi}}$")
    # pm = ax.pcolormesh(grid_x, grid_y, grid_z, shading="nearest")
    c = ax.scatter(
        [x[0] for x in points],
        [x[1] for x in points],
        c=values,
        s=400,
        # norm=matplotlib.colors.LogNorm(),
    )
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Psuedodata Observed Rate")
    fig.savefig(output_path)


def parseAguments():
    parser = argparse.ArgumentParser(description="make plot")
    parser.add_argument("-o", "--output", required=True, type=str, help="")
    parser.add_argument("input")

    return parser.parse_args()


def main():
    args = parseAguments()
    with open(args.input) as f:
        data = json.load(f)

    plotRate(data, args.output)


if __name__ == "__main__":
    main()
