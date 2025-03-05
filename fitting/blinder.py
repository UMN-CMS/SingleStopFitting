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

    def __call__(self, X, Y=None):
        vals = self.vals(X)
        rm = rotationMatrix(self.theta)
        target = self.spread * (rm @ self.sigma) + self.center
        one_point = self.vals(self.normalization_scale * target.unsqueeze(0))

        mask = vals > one_point
        return mask

    @staticmethod
    def fromData(data, spread):
        return makeWindow2D(data, spread)


@dataclass
class MinYCut:
    min_y: float = 0

    def __call__(self, X, Y=None):
        return Y > self.min_y
