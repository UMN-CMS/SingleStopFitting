import json

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from scipy.interpolate import griddata


def main():
    mplhep.style.use("CMS")
    with open("test.json", "r") as f:
        data = json.load(f)
    xyz = np.array([
        #[x["meta"]["mass_stop"], x["meta"]["mass_chargino"], x["data"][2][1]]
        [x["meta"]["mass_stop"], x["meta"]["mass_chargino"], x["data"]]
        for x in data.values()
    ])
    # points = [[x,y] for x,y,z in xyz]
    # values = [z for x,y,z in xyz]
    # print(points)

    # grid_x, grid_y = np.mgrid[1000:2000:25, 100:2000:25]
    # mask = grid_x > grid_y
    # print(mask.shape)
    #grid_x,grid_y = grid_x[mask], grid_y[mask]
    # grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    # grid_x,grid_y,grid_z= grid_x[mask], grid_y[mask], grid_z[mask]
    # grid_z[~mask]  = np.nan
    fig,ax=plt.subplots()
    # pm = ax.pcolormesh(grid_x, grid_y, grid_z)
    pm = ax.scatter(xyz[:,0], xyz[:,1], c=xyz[:,2], s=200)

    
    ax.set_xlabel("$m_{\\tilde{t}}$")
    ax.set_ylabel("$m_{\\tilde{\chi}}$")
    ax.set_ylabel("$m_{\\tilde{\chi}}$")

    mplhep.cms.lumitext(text="2018", ax=ax)
    a, b, c = mplhep.cms.text(text="\nQCD Simulation", ax=ax, loc=2)

    cbar = fig.colorbar(pm, ax=ax)
    cbar.set_label("Expected Significance")
    fig.savefig("test.pdf")


if __name__ == "__main__":
    main()
