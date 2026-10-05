r"""Base class for a loss function."""

__all__ = [
    # Protocols
    "Penalty",
    "SequencePenalty",
    "Loss",
    "SequenceLoss",
    "TimeSeriesLoss",
    # classes
    "SeparableLoss",
    "SeparableSequenceLoss",
    # ABCs
    "BaseLoss",
    "BaseSequenceLoss",
    "Dim",
]

from abc import abstractmethod
from typing import Final, Protocol

import torch
from torch import Tensor, nn

from .reductions import Reduction

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


class BaseLoss(nn.Module, Loss):
    r"""Base class for a sample-wise loss function."""

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
        sample_weights: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        r"""Compute the loss."""
        raise NotImplementedError


class BaseSequenceLoss(nn.Module, SequenceLoss):
    r"""Base class for a time-series losses.

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
        self.combined_dim = (self.time_dim, *self.channel_dim)

        if not {self.time_dim}.isdisjoint(self.channel_dim):
            raise ValueError("Time and channel axes must be disjoint!")

    @abstractmethod
    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., $N, *D]
        targets: Tensor,  # Float[..., $N, *D]
        mask: Tensor | None = None,  # Bool[..., $N, *D]
        sample_weights: Tensor | None = None,  # Float[...]
    ) -> Tensor:  # Float[()]
        r"""Compute the loss."""
        raise NotImplementedError


class SeparableLoss(nn.Module, Loss):
    r"""Base class for separable loss function with reduction over samples.

    Note:
        A loss is separable if there is no interaction between batch elements,
        i.e. $ℒ(x̂, x) = Reduce(ℓ(x̂ₙ, xₙ)ₙ})$ for some per sample loss ℓ
        and a reduction function Reduce.
    """

    penalty: Penalty
    r"""CONST: The penalty function to apply."""
    reduction: Reduction
    r"""CONST: The reduction to apply over the samples."""

    def __init__(
        self,
        *,
        feature_dim: int | tuple[int, ...],
    ) -> None:
        super().__init__()
        self.feature_dim = (
            (feature_dim,) if isinstance(feature_dim, int) else feature_dim
        )

    def forward(
        self,
        *,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        errors = self.penalty(
            predictions=predictions,
            targets=targets,
            mask=mask,
            feature_dim=self.feature_dim,
        )
        return self.reduction.apply(errors, weights=sample_weights)


class SeparableSequenceLoss(nn.Module, SequenceLoss):
    r"""Base class for separable time-series loss with a reduction over samples.

    Note:
        A loss is separable if there is no interaction between batch elements,
        i.e. $ℒ(x̂, x) = Reduce(ℓ(x̂ₙ, xₙ)ₙ})$ for some per sample loss ℓ
        and a reduction function Reduce.
    """

    penalty: SequencePenalty
    r"""CONST: The penalty function to apply per sequence pair."""
    reduction: Reduction
    r"""CONST: The reduction to apply over the samples."""
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

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "penalty"):
            raise TypeError(
                f"Subclasses of {cls.__name__} must define a `penalty` method or class attribute."
            )

    def forward(
        self,
        *,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        errors = self.penalty(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )
        return self.reduction.apply(errors, weights=sample_weights)
