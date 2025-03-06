import json

import matplotlib.pyplot as plt
import mplhep
import numpy as np


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

    fig.colorbar(pm, ax=ax)
    fig.savefig("test.pdf")


if __name__ == "__main__":
    main()
