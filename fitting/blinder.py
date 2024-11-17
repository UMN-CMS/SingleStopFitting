import pickle as pkl
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
from matplotlib.patches import Polygon

from . import regression
from .plot_tools import (
    addAxesToHist,
    getPolyFromSquares,
    makeSquares,
    plotData,
    plotRaw,
)


def rotationMatrix(theta):
    return torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )


def gaussian2D(X, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = X[..., 0], X[..., 1]
    a = (torch.cos(theta) ** 2) / (2 * sigma_x**2) + (torch.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(torch.sin(2 * theta)) / (4 * sigma_x**2) + (torch.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (torch.sin(theta) ** 2) / (2 * sigma_x**2) + (torch.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = amplitude * torch.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g


def gaussian1D(X, amplitude, xo, sigma_x):
    g = amplitude * torch.exp(-(((X - xo) / sigma_x) ** 2))
    return g.ravel()


def numpyGaussian2D(X, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = X[..., 0], X[..., 1]
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g


def numpyGaussian1D(X, amplitude, xo, sigma_x):
    g = amplitude * np.exp(-(((X - xo) / sigma_x) ** 2))
    return g.ravel()


@dataclass
class GaussianWindow1D:
    amplitude: float
    center: float
    sigma: float

    spread: float

    normalization_scale: torch.Tensor

    def __call__(self, X):
        vals = self.vals(X)
        one_point = self.vals(
            self.normalization_scale * (self.center + self.spread * self.sigma)
        )
        mask = vals > one_point
        return mask

    def vals(self, X):
        X = X / self.normalization_scale
        v = gaussian1D(X, self.amplitude, self.center, self.sigma)
        return v


@dataclass
class GaussianWindow2D:
    amplitude: float
    center: torch.Tensor
    sigma: torch.Tensor
    theta: float

    spread: float

    normalization_scale: torch.Tensor

    def vals(self, X):
        X = X / self.normalization_scale
        v = gaussian2D(X, self.amplitude, *self.center, *self.sigma, self.theta)
        return v

    def __call__(self, X):
        vals = self.vals(X)
        rm = rotationMatrix(self.theta)
        target = self.spread * (rm @ self.sigma) + self.center
        one_point = self.vals(self.normalization_scale * target.unsqueeze(0))

        mask = vals > one_point
        return mask


def makeWindow1D(signal_data, spread=1.0):
    X = signal_data.X
    s = X.max(dim=0).values
    X = X / s
    Y = signal_data.Y
    popt, pcov = scipy.optimize.curve_fit(numpyGaussian1D, X, Y)
    return GaussianWindow1D(
        *[torch.Tensor([x]) for x in popt], torch.tensor([spread]), s
    )


def makeWindow2D(signal_data, spread=1.0):
    X = signal_data.X
    s = X.max(dim=0).values
    X = X / s
    Y = signal_data.Y
    popt, pcov = scipy.optimize.curve_fit(
        numpyGaussian2D,
        X,
        Y,
        p0=[1.0, *X[torch.argmax(signal_data.Y)], 0.05, 0.05, 0.0],
    )

    return GaussianWindow2D(
        torch.tensor(popt[0]),
        torch.tensor(popt[1:3]),
        torch.tensor(popt[3:5]),
        torch.tensor([popt[5]]),
        spread,
        s,
    )


def windowPlot1D(signal_data, window, frac=None):
    fig, ax = plt.subplots()
    plotData(ax, signal_data)
    if window.center - window.axes / 2:
        ax.axvline(window.center - window.axes, 0, 1, color="black")
    if window.center + window.axes / 2:
        ax.axvline(window.center + window.axes, 0, 1, color="black")
    return fig, ax


def windowPlots2D(signal_data, window, frac=None):
    fig, ax = plt.subplots()
    ret = {}
    plotData(ax, signal_data)
    if window is not None:
        mask = window(signal_data.X)
        squares = makeSquares(signal_data.X[mask], signal_data.E)
        points = getPolyFromSquares(squares)
        poly = Polygon(points, edgecolor="green", linewidth=3, fill=False)
        ax.add_patch(poly)

        figm, axm = plt.subplots()
        plotRaw(axm, signal_data.E, signal_data.X, mask.to(torch.float64))
        ret["mask"] = (figm, axm)

    ret["sig_window"] = (fig, ax)

    return ret
