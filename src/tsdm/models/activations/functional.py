r"""Implementations of activation functions.

Notes:
    Contains activations in functional form.
    See `tsdm.models.activations.modular` for modular implementations.
"""

__all__ = ["Activation"]

from torch import Tensor
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Activation(Protocol):
    """Protocol for activation functions."""

    def __call__(self, x: Tensor, /) -> Tensor:
        """Apply the activation function."""
        ...
