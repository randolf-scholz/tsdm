r"""Base class for a loss function."""

__all__ = [
    # Protocols
    "Metric",
    "SequentialMetric",
    "IndexedMetric",
    # ABCs
    "BaseMetric",
    "SequentialBaseMetric",
]

from abc import abstractmethod
from typing import Final, Protocol

import torch
from torch import Tensor, nn

type Dim = int | tuple[int, ...] | None


class Metric(Protocol):
    r"""Protocol for metrics on (non-sequential) data.

    .. math:: ℓ：Y×Y ⟶ ℝ_{≥0}
    """

    # (..., *d), (..., *d) -> (...)
    def __call__(self, *, predictions: Tensor, targets: Tensor) -> Tensor: ...


class SequentialMetric(Protocol):
    r"""Protocol for a loss function on sequences of variable length.

    .. math:: ℓ： \Seq(Y) ×_ℕ \Seq(Y) ⟶ ℝ_{≥0}

    Where $\Seq(Y)$ denotes the set of all finite sequences over $Y$

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
    ) -> Tensor:  # Float[...]
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


class IndexedMetric(Protocol):
    r"""Protocol for a loss function on sequences of variable length.

    Similar to `SequentialMetric`, but the input is indexed by a time-steps:

    .. math:: ℓ： \Seq(T) ×_ℕ \Seq(Y) ×_ℕ \Seq(Y) ⟶ ℝ_{≥0}

    This allows for instance for discounting values depending on the time-steps.

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
    ) -> Tensor: ...


class BaseMetric(nn.Module, Metric):
    r"""Base class for a sample-wise loss function."""

    channel_weight: Tensor | None
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
        channel_weight: Tensor | None = None,
        scaled: bool = False,
    ) -> None:
        super().__init__()

        if channel_weight is not None:
            w = torch.as_tensor(channel_weight, dtype=torch.float32)
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

        self.scaled = bool(scaled)
        self.dim = (dim,) if isinstance(dim, int) else tuple(dim)
        self.register_parameter("weight", w)

    @abstractmethod
    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        weight: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        r"""Compute the loss."""
        raise NotImplementedError


class SequentialBaseMetric(nn.Module, SequentialMetric):
    r"""Base class for a time-series function.

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

    channel_weights: Tensor | None
    r"""CONST: Optional weight-tensor for the channel-axes."""
    channel_dim: Final[tuple[int, ...]]
    r"""CONST: The channel-axes over which the loss is computed."""

    combined_dim: Final[tuple[int, ...]]
    r"""CONST: The combined time- and channel-axes."""

    normalize_time: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    normalize_channels: Final[bool]
    r"""CONST: Whether to normalize the weights."""

    # TODO: implement discount factors.

    def __init__(
        self,
        /,
        *,
        time_dim: int = -2,
        channel_dim: Dim = None,
        weight: Tensor | None = None,
        normalize_time: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        if weight is not None:
            w = torch.as_tensor(weight, dtype=torch.float32)
            w = nn.Parameter(w, requires_grad=False)
            dim = (
                tuple(range(-w.ndim, 0))
                if channel_dim is None
                else (channel_dim,)
                if isinstance(channel_dim, int)
                else channel_dim
            )

            if len(dim) != w.ndim:
                raise ValueError(
                    f"Weight tensor has {w.ndim} axes, "
                    f"but axis={dim} specifies {len(dim)} axes."
                )
        else:
            w = None
            dim = (
                (-1,)
                if channel_dim is None
                else (channel_dim,)
                if isinstance(channel_dim, int)
                else channel_dim
            )

        self.register_parameter("channel_weight", w)
        self.time_dim = time_dim
        self.channel_dim = dim
        self.normalize_channels = bool(normalize)
        self.normalize_time = bool(normalize_time)

        self.combined_dim = (self.time_dim, *self.channel_dim)

        if not {self.time_dim}.isdisjoint(self.channel_dim):
            raise ValueError("Time and channel axes must be disjoint!")

    @abstractmethod
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError
