r"""Pretrained Models.

Each model comes with several components:

- the model itself
- the encoder ?
"""

__all__ = [
    # Abstract Base Classes
    "PreTrainedMetaClass",
    "PreTrainedModel",
    # Classes
    "LinODEnet",
]

from typing import Final, TypeAlias

import torch

from tsdm.models.pretrained.base import PreTrainedMetaClass, PreTrainedModel
from tsdm.models.pretrained.linodenet import LinODEnet

Model: TypeAlias = torch.nn.Module
r"""Type hint for models."""

MODELS: Final[dict[str, type[Model]]] = {
    "LinODEnet": LinODEnet,
}
r"""Dictionary of all available models."""
