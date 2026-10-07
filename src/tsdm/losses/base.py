r"""Base class for a loss function."""

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
        feature_dim: int | tuple[int, ...],  # *D
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
    r"""Protocol for losses on arbitrary data.

    .. math:: ℓ：Y×Y ⟶ ℝ_{≥0}
    """

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., *D]
        targets: Tensor,  # Float[..., *D]
    ) -> Tensor: ...  # Float[()]


class SequenceLoss(Protocol):
    r"""Protocol for a loss function on sequences of variable length.

    .. math:: ℓ： \Seq(Y) ×_ℕ \Seq(Y) ⟶ ℝ_{≥0}

    Note:
        $\Seq(Y)$ denotes the set of all finite sequences over $Y$

        .. math:: \Seq(Y) ≔ ∐_{n∈ℕ} Yⁿ ≙ ⋃_{n∈ℕ} Yⁿ

        and the fiber product $×_ℕ$ denotes the set of all pairs of same length sequences:

        .. math:: \Seq(U) ×_ℕ \Seq(V) ≔ {(u, v) ∈ \Seq(U) × \Seq(V) : \abs{u} = \abs{v}}

    Remark:
        The categorical coproduct $∐_{n∈ℕ}Xⁿ$ is used to formally define the finite
        sequence space. As a set, this is just identical to the union $⋃_{n∈ℕ} Xⁿ$.
        However, if $X$ has additional structure, such as being a topological vector
        space, then this induces structure on the finite sequence space.

    Args:
        predictions: Predictions of shape (..., *d), possibly padded NaN.
        targets: Targets of shape (..., *d), possibly padded NaN.
        mask: Mask of shape (..., *d), marking valid values.
    """

    def __call__(
        self,
        *,
        predictions: Tensor,  # Float[..., $N, *D], possibly padded NaN
        targets: Tensor,  # Float[..., $N, *D], possibly padded NaN
        mask: Tensor | None = None,  # Bool[..., $N, *D]
    ) -> Tensor:  # Float[()]
        r"""Compute a loss between the predictions and the targets.

        A time series loss function acts on sequences of variable length.
        Given a collection of pairs of sequences $(x_n,x̂_n)∈⋃_{T∈ℕ}(V⊕V)^T$,
        returns a single scalar. Each pair $(x_n,x̂_n)$ is of equal length $T_n$,
        but different pairs may have different lengths.

        In principle, this means that nested/ragged tensors are required.
        However, for the sake of simplicity, we assume that the tensors are
        padded with missing values, such that they are of equal length.
        """
        ...


class TimeSeriesLoss(Protocol):
    r"""Protocol for a loss function on sequences of variable length with time information.

    .. math:: ℓ： \Seq(T) ×_ℕ \Seq(Y) ×_ℕ \Seq(Y) ⟶ ℝ_{≥0}

    Suitable for continuous-time objectives whose weighting or
    discounting depends on observation/prediction times.

    Args:
        predictions: Predictions of shape (..., *d), possibly padded NaN.
        targets: Targets of shape (..., *d), possibly padded NaN.
        time_steps: Time steps of shape (..., *d), possibly padded NaN.
        mask: Mask of shape (..., *d), marking valid values.
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
    r"""Reduction methods for seperable losses."""

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
        r"""Apply an optional sample weight and outer reduction to losses."""
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
    r"""PARAM: Optional weight-vector."""

    # Constants
    dim: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    scaled: Final[bool]
    r"""CONST: Whether to normalize the channel axis."""

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
    r"""Base class for a sequence losses.

    Because the loss is computed over a sequence of variable length, the default is to normalize
    the loss by the sequence length, so that loss values are comparable across sequences.
    This class can be used to express decomposable losses of the form

    .. math:: 𝓛(x̂，x) ≔ ℓ(x̂ₜ, xₜ)

    By default, the aggregation $𝐀_t$ is the mean over the time-axes $𝐄_t$, but simply
    summing over the time-axes is also possible.
    """

    # Constants
    time_dim: Final[int]
    r"""CONST: The time-axes over which the loss is computed."""
    channel_dim: Final[tuple[int, ...]]
    r"""CONST: The channel-axes over which the loss is computed."""

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
