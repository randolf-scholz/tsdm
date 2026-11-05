r"""Utilities for logging."""

__all__ = [
    "TargetsAndPredictions",
    "Loss",
]


from typing import NamedTuple, Protocol

from torch import Tensor


class TargetsAndPredictions(NamedTuple):
    r"""Pair target and prediction tensors returned by model evaluation."""

    targets: Tensor
    predictions: Tensor


class Loss(Protocol):
    r"""Protocol for scalar losses on target and prediction tensors.

    $$ ℓ：Y×Y ⟶ ℝ_{≥0} $$

    Implementations accept keyword-only `predictions` and `targets` and return
    a scalar tensor. Nonnegativity is a convention, not a runtime check.
    """

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., *D]
        targets: Tensor,  # Float[..., *D]
    ) -> Tensor: ...  # Float[()]
