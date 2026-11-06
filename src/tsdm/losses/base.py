r"""Protocols, reductions, and base classes for tensor and sequence losses."""

__all__ = [
    # Protocols
    "Penalty",
    "SequencePenalty",
    "Loss",
    "SequenceLoss",
    "TimeSeriesLoss",
    # classes
    "Reduction",
    # ABCs
    "BaseLoss",
    "BaseSequenceLoss",
    "Dim",
]

from abc import abstractmethod
from enum import StrEnum
from typing import Final, Protocol, assert_never

import torch
from torch import Tensor, nn

type Dim = int | tuple[int, ...] | None


class Penalty(Protocol):
    r"""Protocol for a penalty term on (non-sequence) data."""

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., *D]
        targets: Tensor,  # Float[..., *D]
        dim: int | tuple[int, ...],  # *D
        mask: Tensor | None = None,  # Bool[...,  *D]
    ) -> Tensor: ...  # Float[...]


class SequencePenalty(Protocol):
    r"""Protocol for a penalty on equal length sequences."""

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., $N, *D]
        targets: Tensor,  # Float[..., $N, *D]
        time_dim: int,  # $N
        channel_dim: int | tuple[int, ...],  # *D
        mask: Tensor | None = None,  # Bool[..., $N, *D]
    ) -> Tensor: ...  # Float[...]


class Loss(Protocol):
    r"""Protocol for scalar losses on target and prediction tensors.

    $$ ℓ：Y×Y ⟶ ℝ_{≥0} $$

    Implementations accept keyword-only `predictions` and `targets` and return a
    scalar tensor. Nonnegativity is a convention of the loss interface.
    """

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., *D]
        targets: Tensor,  # Float[..., *D]
    ) -> Tensor: ...  # Float[()]


class SequenceLoss(Protocol):
    r"""Protocol for scalar losses on pairs of variable-length sequences.

    $$ ℓ：\Seq(Y) ×_ℕ \Seq(Y) ⟶ ℝ_{≥0} $$

    The finite sequence space is the disjoint union

    $$ \Seq(Y) ≔ ∐_{n∈ℕ}Yⁿ $$

    The fiber product pairs $u∈\Seq(U)$ and $v∈\Seq(V)$ of equal length:

    $$ \Seq(U) ×_ℕ \Seq(V) ≔ \{(u,v): |u|=|v|\} $$

    The disjoint union retains the length of each sequence. Additional structure on
    the component spaces can be carried to the sequence space through a coproduct.
    Batches use padded tensors and an optional mask to represent variable lengths.
    """

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., $N, *D], possibly padded NaN
        targets: Tensor,  # Float[..., $N, *D], possibly padded NaN
        mask: Tensor | None = None,  # Bool[..., $N, *D]
    ) -> Tensor:  # Float[()]
        r"""Compute a scalar loss over paired sequences of variable length.

        Each prediction-target pair has the same length, while different pairs may
        have different lengths. Use NaN padding and an explicit validity mask to
        represent ragged batches in dense tensors. Implementations determine how
        missing values are handled when a mask is omitted.

        Args:
            predictions: Predicted sequences of shape `(..., $N, *D)`.
            targets: Target sequences of shape `(..., $N, *D)`.
            mask: Boolean tensor marking valid entries, with the same shape as the data.
        """
        ...


class TimeSeriesLoss(Protocol):
    r"""Protocol for variable-length sequence losses with observation times.

    $$ ℓ：\Seq(T) ×_ℕ \Seq(Y) ×_ℕ \Seq(Y) ⟶ ℝ_{≥0} $$

    The fiber products require time, prediction, and target sequences to have equal
    lengths within each sample. This interface supports continuous-time objectives
    whose weighting or discounting depends on observation or prediction times.

    Predictions and targets have shape `(..., $N, *D)`; `time_steps` has shape
    `(..., $N)`. These tensors may use NaN padding. The optional boolean `mask`
    has the data shape and marks valid entries. Implementations return a scalar tensor.
    """

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., $N, *D], possibly padded NaN
        targets: Tensor,  # Float[..., $N, *D], possibly padded NaN
        time_steps: Tensor,  # Float[..., $N], possibly padded NaN
        mask: Tensor | None = None,  # Bool[..., $N, *D]
    ) -> Tensor: ...  # Float[()]


class Reduction(StrEnum):
    r"""Reduction methods for separable losses."""

    SUM = "sum"
    MEAN = "mean"
    NONE = "none"

    def apply(
        self,
        values: Tensor,  # Float[...]
        /,
        *,
        weights: Tensor | None = None,  # Float[...]
    ) -> Tensor:  # Float[()]
        r"""Apply optional sample weights and reduce per-sample losses.

        Weights must broadcast to `values.shape`. `NONE` returns the element-wise
        weighted values, `SUM` sums them, and `MEAN` divides their sum by the sum of
        the broadcast weights. Without weights, use the ordinary sum or mean.
        Zero total weight is not guarded against and can produce NaN or infinity.
        """
        if weights is not None:
            weights = torch.broadcast_to(weights, values.shape)

        match self:
            case Reduction.NONE:
                return values if weights is None else weights * values
            case Reduction.SUM:
                return (
                    values.sum()
                    if weights is None
                    else torch.dot(weights.flatten(), values.flatten())
                )
            case Reduction.MEAN:
                return (
                    values.mean()
                    if weights is None
                    else torch.dot(weights.flatten(), values.flatten()) / weights.sum()
                )
            case other:
                assert_never(other)


class BaseLoss(nn.Module, Loss):
    r"""Base class for loss functions."""

    feature_weights: Tensor | None
    r"""Optional nontrainable feature-weight tensor."""

    # Constants
    dim: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    scaled: Final[bool]
    r"""CONST: Whether to normalize powered sums by the number of active features."""

    def __init__(
        self,
        *,
        dim: Dim = None,
        feature_weights: Tensor | None = None,
        scaled: bool = False,
    ) -> None:
        super().__init__()

        if feature_weights is not None:
            w = torch.as_tensor(feature_weights, dtype=torch.float32)
            w = nn.Parameter(w, requires_grad=False)
            dim = (
                tuple(range(-w.ndim, 0))
                if dim is None
                else (dim,)
                if isinstance(dim, int)
                else dim
            )

            if len(dim) != w.ndim:
                raise ValueError(
                    f"Weight tensor has {w.ndim} axes, "
                    f"but axis={dim} specifies {len(dim)} axes."
                )
        else:
            w = None
            dim = -1 if dim is None else dim

        self.scaled = scaled
        self.dim = (dim,) if isinstance(dim, int) else tuple(dim)
        self.feature_weights = w

    @abstractmethod
    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
    ) -> Tensor:  # Float[()]
        r"""Compute the loss."""
        raise NotImplementedError


class BaseSequenceLoss(nn.Module, SequenceLoss):
    r"""Base class storing time and channel axes for sequence losses.

    Concrete losses specify how to aggregate per-time penalties, for example

    $$ 𝓛(x̂,x) ≔ 𝐀ₜ\bigl(ℓ(x̂ₜ，xₜ)\bigr) $$

    Here $𝐀ₜ$ is an aggregation operator like the a mean or sum over time.
    Length normalization can make losses comparable across sequences of different lengths.
    This base class stores axis selections and checks that the supplied time axis is not among
    the channel axes; subclasses implement the loss and its normalization.
    """

    # Constants
    time_dim: Final[int]
    r"""CONST: The time axis over which the loss is computed."""
    channel_dim: Final[tuple[int, ...]]
    r"""CONST: The channel axes over which the loss is computed."""

    def __init__(self, *, time_dim: int, channel_dim: int | tuple[int, ...]) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.channel_dim = (
            (channel_dim,) if isinstance(channel_dim, int) else channel_dim
        )
        if not {self.time_dim}.isdisjoint(self.channel_dim):
            raise ValueError("Time and channel axes must be disjoint!")

    @abstractmethod
    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., $N, *D]
        targets: Tensor,  # Float[..., $N, *D]
        mask: Tensor | None = None,  # Bool[..., $N, *D]
    ) -> Tensor:  # Float[()]
        r"""Compute the loss."""
        raise NotImplementedError
