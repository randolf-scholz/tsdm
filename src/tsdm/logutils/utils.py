r"""Utilities for logging."""

__all__ = [
    "TargetsAndPredictions",
    "Loss",
]


from typing import NamedTuple, Protocol

from torch import Tensor


class TargetsAndPredictions(NamedTuple):
    r"""Targets and predictions."""

    targets: Tensor
    predictions: Tensor


class Loss(Protocol):
    r"""Protocol for losses on arbitrary data.

    .. math:: ℓ：Y×Y ⟶ ℝ_{≥0}
    """

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., *D]
        targets: Tensor,  # Float[..., *D]
    ) -> Tensor: ...  # Float[()]
