from typing import Annotated
import numpy as np
import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    ConfigDict,
    TypeAdapter,
    Field,
)


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
