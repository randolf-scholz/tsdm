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


from tsdm.models.pretrained.base import PreTrained, PreTrainedBase
from tsdm.models.pretrained.linodenet import LinODEnet

MODELS: dict[str, type[PreTrainedBase]] = {
    # "OldLinODEnet": OldLinODEnet,
    "LinODEnet": LinODEnet,
}
r"""Dictionary of all available models."""
