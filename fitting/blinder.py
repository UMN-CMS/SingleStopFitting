from pydantic import BaseModel, ConfigDict
import json
from .core import TorchTensor, Metadata

import numpy as np
import torch


def rotationMatrix(theta):
    return torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )


def numpyGaussian2D(X, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = X[..., 0], X[..., 1]
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
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
    import scipy.optimize

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
        amplitude=torch.tensor(popt[0]),
        center=torch.tensor(popt[1:3]),
        sigma=torch.tensor(popt[3:5]),
        theta=torch.tensor([popt[5]]),
        spread=spread,
        normalization_scale=s,
    )


class GaussianWindow2D(BaseModel):
    amplitude: float
    center: TorchTensor
    sigma: TorchTensor
    theta: TorchTensor

    spread: float

    normalization_scale: TorchTensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class StaticWindow(BaseModel):
    mask: TorchTensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, X, Y=None):
        return self.mask

    @staticmethod
    def fromFile(path):
        mask = []
        with open(path, "r") as f:
            for line in f:
                l = []
                for c in line:
                    l.append(c == "X")
                mask.append(l)
        mask = list(reversed(mask))
        mask = torch.Tensor(mask)
        mask = mask.to(bool)
        mask = torch.flatten(mask)
        return StaticWindow(mask)


class MinYCut(BaseModel):
    min_y: float = 0

    def __call__(self, X, Y=None):
        return Y > self.min_y

Metadata.model_rebuild()
