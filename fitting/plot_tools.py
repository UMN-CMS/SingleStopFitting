from collections import Counter, namedtuple
import torch
from .utils import pointsToGrid, dataToHist
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


def hist2dplot(
    H,
    xbins=None,
    ybins=None,
    labels=None,
    cbar: bool = True,
    cbarsize="7%",
    cbarpad=0.2,
    cbarpos="right",
    cbarextend=True,
    cmin=None,
    cmax=None,
    ax = None,
    flow="hint",
    **kwargs,
):


    import inspect
    import matplotlib.pyplot as plt
    import numpy as np
    from mplhep.plot import append_axes
    from mplhep.utils import (
        get_histogram_axes_title,
        get_plottable_protocol_bins,
        hist_object_handler,
        isLight,
        align_marker,
        to_padded2d,
    )
    from collections import namedtuple

    ColormeshArtists = namedtuple("ColormeshArtists", "pcolormesh cbar text")


    """
    Create a 2D histogram plot from `np.histogram`-like inputs.

    Parameters
    ----------
    H : object
        Histogram object with containing values and optionally bins. Can be:

        - `np.histogram` tuple
        - `boost_histogram` histogram object
        - raw histogram values as list of list or 2d-array

    xbins : 1D array-like, optional, default None
        Histogram bins along x axis, if not part of ``H``.
    ybins : 1D array-like, optional, default None
        Histogram bins along y axis, if not part of ``H``.
    labels : 2D array (H-like) or bool, default None, optional
        Array of per-bin labels to display. If ``True`` will
        display numerical values
    cbar : bool, optional, default True
        Draw a colorbar. In contrast to mpl behaviors the cbar axes is
        appended in such a way that it doesn't modify the original axes
        width:height ratio.
    cbarsize : str or float, optional, default "7%"
        Colorbar width.
    cbarpad : float, optional, default 0.2
        Colorbar distance from main axis.
    cbarpos : {'right', 'left', 'bottom', 'top'}, optional,  default "right"
        Colorbar position w.r.t main axis.
    cbarextend : bool, optional, default False
        Extends figure size to keep original axes size same as without cbar.
        Only safe for 1 axes per fig.
    cmin : float, optional
        Colorbar minimum.
    cmax : float, optional
        Colorbar maximum.
    ax : matplotlib.axes.Axes, optional
        Axes object (if None, last one is fetched or one is created)
    flow :  str, optional {"show", "sum","hint", None}
            Whether plot the under/overflow bin. If "show", add additional under/overflow bin. If "sum", add the under/overflow bin content to first/last bin. "hint" would highlight the bins with under/overflow contents
    **kwargs :
        Keyword arguments passed to underlying matplotlib function - pcolormesh.

    Returns
    -------
        Hist2DArtist

    """

    # ax check
    if ax is None:
        ax = plt.gca()
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")

    h = hist_object_handler(H, xbins, ybins)

    # TODO: use Histogram everywhere

    H = np.copy(h.values())
    xbins, xtick_labels = get_plottable_protocol_bins(h.axes[0])
    ybins, ytick_labels = get_plottable_protocol_bins(h.axes[1])
    # Show under/overflow bins
    # "show": Add additional bin with 2 times bin width
    if (
        hasattr(h, "values")
        and "flow" not in inspect.getfullargspec(h.values).args
        and flow is not None
    ):
        print(
            f"Warning: {type(h)} is not allowed to get flow bins, flow bin option set to None"
        )
        flow = None
    elif flow in ["hint", "show"]:
        xwidth, ywidth = (xbins[-1] - xbins[0]) * 0.05, (ybins[-1] - ybins[0]) * 0.05
        pxbins = np.r_[xbins[0] - xwidth, xbins, xbins[-1] + xwidth]
        pybins = np.r_[ybins[0] - ywidth, ybins, ybins[-1] + ywidth]
        padded = to_padded2d(h)
        hint_xlo, hint_xhi, hint_ylo, hint_yhi = True, True, True, True
        if np.all(padded[0, :] == 0):
            padded = padded[1:, :]
            pxbins = pxbins[1:]
            hint_xlo = False
        if np.all(padded[-1, :] == 0):
            padded = padded[:-1, :]
            pxbins = pxbins[:-1]
            hint_xhi = False
        if np.all(padded[:, 0] == 0):
            padded = padded[:, 1:]
            pybins = pybins[1:]
            hint_ylo = False
        if np.all(padded[:, -1] == 0):
            padded = padded[:, :-1]
            pybins = pybins[:-1]
            hint_yhi = False
        if flow == "show":
            H = padded
            xbins, ybins = pxbins, pybins
    elif flow == "sum":
        H = np.copy(h.values())
        # Sum borders
        try:
            H[0], H[-1] = (
                H[0] + h.values(flow=True)[0, 1:-1],  # type: ignore[call-arg]
                H[-1] + h.values(flow=True)[-1, 1:-1],  # type: ignore[call-arg]
            )
            H[:, 0], H[:, -1] = (
                H[:, 0] + h.values(flow=True)[1:-1, 0],  # type: ignore[call-arg]
                H[:, -1] + h.values(flow=True)[1:-1, -1],  # type: ignore[call-arg]
            )
            # Sum corners to corners
            H[0, 0], H[-1, -1], H[0, -1], H[-1, 0] = (
                h.values(flow=True)[0, 0] + H[0, 0],  # type: ignore[call-arg]
                h.values(flow=True)[-1, -1] + H[-1, -1],  # type: ignore[call-arg]
                h.values(flow=True)[0, -1] + H[0, -1],  # type: ignore[call-arg]
                h.values(flow=True)[-1, 0] + H[-1, 0],  # type: ignore[call-arg]
            )
        except TypeError as error:
            if "got an unexpected keyword argument 'flow'" in str(error):
                raise TypeError(
                    f"The histograms value method {repr(h)} does not take a 'flow' argument. UHI Plottable doesn't require this to have, but it is required for this function."
                    f" Implementations like hist/boost-histogram support this argument."
                ) from error
    xbin_centers = xbins[1:] - np.diff(xbins) / float(2)
    ybin_centers = ybins[1:] - np.diff(ybins) / float(2)

    _x_axes_label = ax.get_xlabel()
    x_axes_label = (
        _x_axes_label if _x_axes_label != "" else get_histogram_axes_title(h.axes[0])
    )
    _y_axes_label = ax.get_ylabel()
    y_axes_label = (
        _y_axes_label if _y_axes_label != "" else get_histogram_axes_title(h.axes[1])
    )

    H = H.T

    # if cmin is not None:
    #     H[H < cmin] = None
    # if cmax is not None:
    #     H[H > cmax] = None

    X, Y = np.meshgrid(xbins, ybins)

    kwargs.setdefault("shading", "flat")
    pc = ax.pcolormesh(X, Y, H, vmin=cmin, vmax=cmax, **kwargs)

    if x_axes_label:
        ax.set_xlabel(x_axes_label)
    if y_axes_label:
        ax.set_ylabel(y_axes_label)

    ax.set_xlim(xbins[0], xbins[-1])
    ax.set_ylim(ybins[0], ybins[-1])

    if xtick_labels is None:  # Ordered axis
        if len(ax.get_xticks()) > len(xbins) * 0.7:
            ax.set_xticks(xbins)
    else:  # Categorical axis
        ax.set_xticks(xbin_centers)
        ax.set_xticklabels(xtick_labels)
    if ytick_labels is None:
        if len(ax.get_yticks()) > len(ybins) * 0.7:
            ax.set_yticks(ybins)
    else:  # Categorical axis
        ax.set_yticks(ybin_centers)
        ax.set_yticklabels(ytick_labels)

    if cbar:
        cax = append_axes(
            ax, size=cbarsize, pad=cbarpad, position=cbarpos, extend=cbarextend
        )
        cb_obj = plt.colorbar(pc, cax=cax)
    else:
        cb_obj = None

    plt.sca(ax)
    if flow == "show":
        if hint_xlo:
            ax.plot(
                [xbins[1]] * 2,
                [0, 1],
                ls="--",
                color="lightgrey",
                clip_on=False,
                transform=ax.get_xaxis_transform(),
            )
        if hint_xhi:
            ax.plot(
                [xbins[-2]] * 2,
                [0, 1],
                ls="--",
                color="lightgrey",
                clip_on=False,
                transform=ax.get_xaxis_transform(),
            )
        if hint_ylo:
            ax.plot(
                [0, 1],
                [ybins[1]] * 2,
                ls="--",
                color="lightgrey",
                clip_on=False,
                transform=ax.get_yaxis_transform(),
            )
        if hint_yhi:
            ax.plot(
                [0, 1],
                [ybins[-2]] * 2,
                ls="--",
                color="lightgrey",
                clip_on=False,
                transform=ax.get_yaxis_transform(),
            )
    elif flow == "hint":
        if (fig := ax.figure) is None:
            raise ValueError("No figure found.")
        _marker_size = (
            30
            * ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        )
        if hint_xlo:
            ax.scatter(
                0,
                0,
                _marker_size,
                marker=align_marker("<", halign="right", valign="bottom"),
                edgecolor="black",
                zorder=5,
                clip_on=False,
                facecolor="white",
                transform=ax.transAxes,
            )
        if hint_xhi:
            ax.scatter(
                1,
                0,
                _marker_size,
                marker=align_marker(">", halign="left"),
                edgecolor="black",
                zorder=5,
                clip_on=False,
                facecolor="white",
                transform=ax.transAxes,
            )
        if hint_ylo:
            ax.scatter(
                0,
                0,
                _marker_size,
                marker=align_marker("v", valign="top", halign="left"),
                edgecolor="black",
                zorder=5,
                clip_on=False,
                facecolor="white",
                transform=ax.transAxes,
            )
        if hint_yhi:
            ax.scatter(
                0,
                1,
                _marker_size,
                marker=align_marker("^", valign="bottom"),
                edgecolor="black",
                zorder=5,
                clip_on=False,
                facecolor="white",
                transform=ax.transAxes,
            )

    _labels: np.ndarray | None = None
    if isinstance(labels, bool):
        _labels = H if labels else None
    elif np.iterable(labels):
        label_array = np.asarray(labels).T
        if H.shape == label_array.shape:
            _labels = label_array
        else:
            raise ValueError(
                f"Labels input has incorrect shape (expect: {H.shape}, got: {label_array.shape})"
            )
    elif labels is not None:
        raise ValueError(
            "Labels not understood, either specify a bool or a Hist-like array"
        )

    text_artists = []
    if _labels is not None:
        if (pccmap := pc.cmap) is None:
            raise ValueError("No colormap found.")
        for ix, xc in enumerate(xbin_centers):
            for iy, yc in enumerate(ybin_centers):
                normedh = pc.norm(H[iy, ix])
                color = "black" if isLight(pccmap(normedh)[:-1]) else "lightgrey"
                text_artists.append(
                    ax.text(
                        xc, yc, _labels[iy, ix], ha="center", va="center", color=color
                    )
                )

    return ColormeshArtists(pc, cb_obj, text_artists)



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
