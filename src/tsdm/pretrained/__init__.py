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
    "LinODEnet",
    # Constants
    "MODELS",
]


from .base import PreTrained, PreTrainedBase
from .linodenet import LinODEnet

MODELS: dict[str, type[PreTrainedBase]] = {
    "LinODEnet": LinODEnet,
}  # fmt: skip
r"""Dictionary of all available models."""
