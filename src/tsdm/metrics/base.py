r"""Base class for a loss function."""

__all__ = [
    # ABCs & Protocols
    "Metric",
    "NN_Metric",
    "BaseMetric",
]

from abc import abstractmethod
from typing import Final, Protocol

import torch
from torch import Tensor, nn

from tsdm.types.aliases import Axis


class Metric(Protocol):
    r"""Represents a metric."""

    def __call__(self, *, predictions: Tensor, targets: Tensor) -> Tensor: ...


class NN_Metric(Protocol):
    r"""Protocol for a loss function."""

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor: ...


class BaseMetric(nn.Module, Metric):
    r"""Base class for a loss function."""

    weight: Tensor | None
    r"""PARAM: Optional weight-vector."""

    # Constants
    axis: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    normalize: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""

    def __init__(
        self,
        /,
        *,
        weight: Tensor | None = None,
        axis: Axis = None,
        normalize: bool = False,
        learnable: bool = False,
    ) -> None:
        super().__init__()

        if weight is not None:
            w = torch.as_tensor(weight, dtype=torch.float32)
            if not torch.all(w >= 0) and torch.any(w > 0):
                raise ValueError(
                    "Weights must be non-negative and at least one must be positive."
                )
            w = nn.Parameter(w / torch.sum(w), requires_grad=self.learnable)
            axis = tuple(range(-w.ndim, 0)) if axis is None else axis
        else:
            w = None
            axis = -1 if axis is None else axis

        self.normalize = normalize
        self.axis = (axis,) if isinstance(axis, int) else tuple(axis)
        self.learnable = bool(learnable)
        self.register_parameter("weight", w)

    @abstractmethod
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError
