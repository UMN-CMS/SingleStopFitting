import torch
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer, ConfigDict


def torchDeserialize(x: list) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    # l = json.load(x)
    return torch.tensor(x)


def torchSerialize(x: torch.Tensor) -> list:
    return x.tolist()


TorchTensor = Annotated[
    torch.Tensor,
    BeforeValidator(torchDeserialize),
    PlainSerializer(torchSerialize, return_type=list),
]


class SignalPoint(BaseModel):
    coupling: str
    mt: float
    mx: float


class FitRegion(BaseModel):
    stop_bounds: TorchTensor
    ratio_bounds: TorchTensor

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FitParams(BaseModel):
    iterations: int
    learning_rate: float
    injected_signal: float


class Metadata(BaseModel):
    signal_point: SignalPoint
    fit_region: FitRegion
    fit_params: FitParams
    window: "GaussianWindow2D"
    other_data: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)
