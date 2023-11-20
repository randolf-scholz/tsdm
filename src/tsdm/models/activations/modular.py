r"""Implementations of activation functions.

Contains activations in modular form.

See Also:
    - `tsdm.models.activations.functional` for functional implementations.
"""

__all__ = ["NN_Activation"]

from torch import Tensor
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class NN_Activation(Protocol):
    """Protocol for activation functions."""

    def forward(self, x: Tensor, /) -> Tensor:
        """Apply the activation function."""
        ...
