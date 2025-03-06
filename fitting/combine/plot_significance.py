import json

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from scipy.interpolate import griddata


def main():
    mplhep.style.use("CMS")
    with open("significance.json", "r") as f:
        data = json.load(f)
    xyz = [
        [x["meta"]["mass_stop"], x["meta"]["mass_chargino"], x["data"]["significance"]]
        for x in data.values()
    ]
    points = [[x,y] for x,y,z in xyz]
    values = [z for x,y,z in xyz]
    print(points)

    grid_x, grid_y = np.mgrid[1000:2000:25, 100:2000:25]
    mask = grid_x > grid_y
    print(mask.shape)
    #grid_x,grid_y = grid_x[mask], grid_y[mask]
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    # grid_x,grid_y,grid_z= grid_x[mask], grid_y[mask], grid_z[mask]
    grid_z[~mask]  = np.nan
    fig,ax=plt.subplots()
    ax.set_xlabel(r'$m_{\tilde{t}}$')
    ax.set_ylabel(r'$m_{\tilde{\chi}}$')
    pm = ax.pcolormesh(grid_x, grid_y, grid_z, shading="nearest")
    ax.scatter([x[0] for x in points], [x[1] for x in points], color="red")
    cb = fig.colorbar(pm, ax=ax)
    cb.set_label(r"$\sigma_{exp}$")
    fig.savefig("expected_signifiances_1sigma_injected.pdf")


if __name__ == "__main__":
    main()
