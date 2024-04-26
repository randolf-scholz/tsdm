"""Base classes for activations."""

__all__ = [
    # ABCs & Protocols
    "Activation",
    "GenericActivation",
    "ActivationABC",
]

from abc import abstractmethod
from collections.abc import Callable

from torch import Tensor, nn
from typing_extensions import (
    Concatenate,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

GenericActivation: TypeAlias = Callable[Concatenate[Tensor, ...], Tensor]
r"""Type alias for generic activation functions (may require additional args!)."""


@runtime_checkable
class Activation(Protocol):
    r"""Protocol for Activation Components."""

    @abstractmethod
    def __call__(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the activation.

        .. Signature: ``[..., n] -> [..., n]``.
        """
        ...


class ActivationABC(nn.Module):
    r"""Abstract Base Class for Activation components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the activation.

        .. Signature: ``... -> ...``.

        Args:
            x: The input tensor to be activated.

        Returns:
            y: The activated tensor.
        """
