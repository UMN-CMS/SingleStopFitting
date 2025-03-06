from collections import Counter, namedtuple
import torch
from ..utils import pointsToGrid, dataToHist
from .patchmpl import hist2dplot
import mplhep

Point = namedtuple("Point", "x y")
Square = namedtuple("Square", "bl s")

import mplhep
def addAxesToHist(ax, size=0.1, pad=0.1, position="bottom", extend=False):
    new_ax = mplhep.append_axes(ax, size, pad, position, extend)
    current_axes = getattr(ax, f"{position}_axes", [])
    setattr(ax, f"{position}_axes", current_axes + [new_ax])
    return new_ax


def makeSquares(points, edges):
    e = torch.meshgrid(edges, indexing="ij")
    e = torch.stack(e, axis=-1)
    e = e[:-1, :-1]
    d = torch.meshgrid(torch.diff(edges[0]), torch.diff(edges[1]), indexing="ij")
    d = torch.stack(d, axis=-1)
    d1 = d.clone()
    d1[..., :] = 0
    d2 = d.clone()
    d2[..., 0] = 0
    d3 = d.clone()
    d3[..., 1] = 0
    e = torch.unsqueeze(e, axis=-2)
    all_diffs = torch.stack((d, d1, d2, d3), axis=-2)
    f = all_diffs + e
    h, _ = pointsToGrid(points, torch.ones_like(points[:, 0]), edges)
    t = f[h.hist > 0]
    return t

def lexsort(keys, dim=-1):
    idx = keys[:,-1].argsort(dim=dim, stable=True)
    for i in range(keys.size(dim)-2, -1, -1):
        k = keys.select(dim, i)
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx

def getPolyFromSquares(squares):
    l = list(
        tuple(round(y, 4) for y in x) for x in torch.flatten(squares, 0, 1).tolist()
    )
    boundary_points = list(x for x, y in Counter(l).items() if y % 2 == 1)
    b = torch.tensor(boundary_points)
    center = b.mean(axis=0)
    diffs = b - center
    angles = torch.atan2(diffs[:, 1], diffs[:, 0]).round(decimals=5)
    stacked = torch.stack((angles, torch.linalg.vector_norm(diffs, dim=-1)), dim=-1)
    a = lexsort(stacked)
    #mask = torch.argsort(angles)
    return b[a].tolist()

def convexHull(points):
    e = 0.001

    def close(a, b):
        return abs(a - b) < e

    def slope(p1, p2):
        if close(p1[0], p2[0]):
            return float("inf")
        else:
            return 1.0 * ((p2[1] - p1[1]) / (p2[0] - p1[0]))

    def cross(p1, p2, p3):
        ret = ((p2[0] - p1[0]) * (p3[1] - p1[1])) - ((p2[1] - p1[1]) * (p3[0] - p1[0]))
        return ret

    stack = sorted(list(set(points)))
    start = stack.pop()
    stack = sorted(stack, key=lambda x: (slope(x, start), -x[1], x[0]))
    hull = [start]
    for p in stack:
        hull.append(p)
        while len(hull) > 2 and cross(hull[-3], hull[-2], hull[-1]) < 0:
            hull.pop(-2)
    return hull




def plotRaw(ax, e, x, y, var=None, **kwargs):
    h = dataToHist(x, y, e, V=var)
    if len(e) == 1:
        DROP_KWARGS = {"cmin", "cmax", "cmap"}
        kwargs = {x:y for x,y in kwargs.items() if x not in DROP_KWARGS}
        return mplhep.histplot(h, ax=ax,  **kwargs)
    elif len(e) == 2:
        return hist2dplot(h, ax=ax, flow=None, **kwargs)


def plotData(ax, data, **kwargs):
    return plotRaw(ax, data.E, data.X, data.Y, var=data.V, **kwargs)
