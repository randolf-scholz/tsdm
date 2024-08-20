r"""Base class for a loss function."""

__all__ = [
    # ABCs & Protocols
    "Metric",
    "NN_Metric",
    "BaseMetric",
    "WeightedMetric",
]

from abc import abstractmethod
from typing import Final, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from tsdm.types.aliases import Axis


@runtime_checkable
class Metric(Protocol):
    r"""Represents a metric."""

    def __call__(self, targets: Tensor, predictions: Tensor, /) -> Tensor:
        r"""Compute the loss."""
        ...


@runtime_checkable
class NN_Metric(Protocol):
    r"""Protocol for a loss function."""

    def forward(self, targets: Tensor, predictions: Tensor, /) -> Tensor:
        r"""Compute the loss."""
        ...


class BaseMetric(nn.Module, Metric):
    r"""Base class for a loss function."""

    # Constants
    axis: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    normalize: Final[bool]
    r"""CONST: Whether to normalize the weights."""

    def __init__(
        self,
        /,
        *,
        axis: int | tuple[int, ...] = -1,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.axis = (axis,) if isinstance(axis, int) else tuple(axis)

    @abstractmethod
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


class WeightedMetric(BaseMetric, Metric):
    r"""Base class for a weighted loss function."""

    # Parameters
    weight: Tensor
    r"""PARAM: The weight-vector."""

    # Constants
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""

    def __init__(
        self,
        weight: Tensor,
        /,
        *,
        learnable: bool = False,
        normalize: bool = False,
        axis: Axis = None,
    ) -> None:
        r"""Initialize the loss function."""
        w = torch.as_tensor(weight, dtype=torch.float32)
        if not torch.all(w >= 0) and torch.any(w > 0):
            raise ValueError(
                "Weights must be non-negative and at least one must be positive."
            )
        axis = tuple(range(-w.ndim, 0)) if axis is None else axis
        super().__init__(axis=axis, normalize=normalize)

        # Set the weight tensor.
        w = w / torch.sum(w)
        self.weight = nn.Parameter(w, requires_grad=self.learnable)
        self.learnable = learnable

        # Validate the axes.
        if len(self.axis) != self.weight.ndim:
            raise ValueError(
                "Number of axes does not match weight shape:"
                f" {len(self.axis)} != {self.weight.ndim=}"
            )

    @abstractmethod
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError
