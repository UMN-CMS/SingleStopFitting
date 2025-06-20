import torch
from typing import Annotated, Any
from fitting.blinder import GaussianWindow2D
from fitting.extra_types import TorchTensor

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    ConfigDict,
    TypeAdapter,
    Field,
)


class SignalPoint(BaseModel, frozen=True):
    coupling: str
    mt: float
    mx: float

    def __lt__(self, other):
        return (self.coupling, self.mt, self.mx) < (other.coupling, other.mt, other.mx)

    def __eq__(self, other):
        return (self.coupling, self.mt, self.mx) == (other.coupling, other.mt, other.mx)


class FitRegion(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stop_bounds: TorchTensor
    ratio_bounds: TorchTensor

    background_toy: int | None = None

    def __eq__(self, other):
        return (self.stop_bounds, self.ratio_bounds, self.background_toy) == (
            other.stop_bounds,
            other.ratio_bounds,
            other.background_toy,
        )


class FitParams(BaseModel):
    iterations: int
    learning_rate: float
    injected_signal: float


class Metadata(BaseModel):
    signal_point: SignalPoint
    fit_region: FitRegion
    fit_params: FitParams
    window: GaussianWindow2D
    other_data: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SignalRun(BaseModel):
    metadata: Metadata
    chi2_info: dict
    post_pred_info: dict
    regression_plots: dict[str, str] = Field(default_factory=dict)
    inference_data: dict[str, Any] = Field(default_factory=dict)

    @property
    def signal_point(self):
        return self.metadata.signal_point

    @property
    def signal_injected(self):
        return self.metadata.fit_params.injected_signal


signal_run_list_adapter = TypeAdapter(list[SignalRun])
signal_run_dict_adapter = TypeAdapter(dict[SignalPoint, list[SignalRun]])
