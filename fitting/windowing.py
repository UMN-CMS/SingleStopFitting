from dataclasses import asdict, dataclass
from typing import Tuple

import torch


@dataclass
class RectWindow:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

    def __call__(self, points, X, Y):
        a = (X > self.x_range[0]) & (X < self.x_range[1])
        b = (Y > self.y_range[0]) & (Y < self.y_range[1])
        return a & b

    def toString(self):
        return f"E__{self.x_range[0]}_{self.x_range[1]}_{self.y_range[0]}_{self.y_range[1]}".replace(
            ".", "p"
        )

    def toDict(self):
        return asdict(self)


@dataclass
class EllipseWindow:
    center: torch.Tensor
    axes: torch.Tensor

    def __call__(self, *vals):
        stacked = torch.stack(vals, axis=-1)
        rel = ((stacked - self.center) ** 2) / self.axes**2
        mask = torch.sum(rel, axis=-1) <= 1.0
        return mask

    def toDict(self):
        return asdict(self)
