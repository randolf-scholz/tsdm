r"""Pretrained Models.

Each model comes with several components:

- the model itself
- the encoder ?
"""

__all__ = [
    # Abstract Base Classes
    "PreTrainedMeta",
    "PreTrainedBase",
    # Classes
    # "OldLinODEnet",
    "LinODEnet",
    # Encoders
    # "LinODEnetEncoder",
]

from typing import Final, TypeAlias

from torch import nn

from tsdm.models.pretrained.base import PreTrainedBase, PreTrainedMeta
from tsdm.models.pretrained.linodenet import LinODEnet

Model: TypeAlias = PreTrainedBase
r"""Type hint for models."""

MODELS: Final[dict[str, type[Model]]] = {
    # "OldLinODEnet": OldLinODEnet,
    "LinODEnet": LinODEnet,
}
r"""Dictionary of all available models."""

del Final, TypeAlias, nn
