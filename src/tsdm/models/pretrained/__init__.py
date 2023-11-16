r"""Pretrained Models.

Each model comes with several components:

- the model itself
- the encoder ?
"""

__all__ = [
    # Protocol
    "PreTrained",
    # Abstract Base Classes
    "PreTrainedBase",
    # Classes
    # "OldLinODEnet",
    "LinODEnet",
    # Encoders
    # "LinODEnetEncoder",
    # Types
    "Model",
    # Constants
    "MODELS",
]

from torch import nn
from typing_extensions import Final, TypeAlias

from tsdm.models.pretrained.base import PreTrained, PreTrainedBase
from tsdm.models.pretrained.linodenet import LinODEnet

Model: TypeAlias = PreTrainedBase
r"""Type hint for models."""

MODELS: Final[dict[str, type[Model]]] = {
    # "OldLinODEnet": OldLinODEnet,
    "LinODEnet": LinODEnet,
}
r"""Dictionary of all available models."""

del Final, TypeAlias, nn
