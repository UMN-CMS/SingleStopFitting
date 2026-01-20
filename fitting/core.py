import torch
from typing import Annotated, Any, Callable
from fitting.blinder import GaussianWindow2D
from fitting.extra_types import TorchTensor
from collections import defaultdict

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    ConfigDict,
    TypeAdapter,
    Field,
    RootModel,
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
    window: GaussianWindow2D | None
    other_data: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def year(self):
        return self.other_data["signal_params"]["dataset"]["era"]["name"]


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


class SignalRunCollection(RootModel):
    root: list[SignalRun]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)

    def append(self, item):
        return self.root.append(item)

    def getProps(self, getter):
        return [getter(x) for x in self.runs]

    def filter(
        self,
        signal_id: SignalPoint | None = None,
        background_toy: int | None = None,
        signal_injected: float | None = None,
        spread: float | None = None,
        year: str | None = None,
        other_filter: Callable[[SignalRun], bool] | None = None,
        key=lambda x: x.signal_point,
    ):
        def f(item):
            ret = True
            if signal_id is not None:
                ret = ret and item.signal_point == signal_id
            if background_toy is not None:
                ret = ret and item.metadata.fit_region.background_toy == background_toy
            if signal_injected is not None:
                ret = ret and item.signal_injected == signal_injected
            if spread is not None:
                ret = ret and item.metadata.window.spread == spread
            if year is not None:
                ret = ret and item.metadata.year == year
            if other_filter is not None:
                ret = ret and other_filter(item)
            return ret

        return SignalRunCollection(sorted((x for x in self.root if f(x)), key=key))

    def groupby(
        self,
        key=lambda x: x.signal_point,
    ):
        ret = defaultdict(lambda: SignalRunCollection([]))
        for run in self.root:
            ret[key(run)].append(run)
        return ret
