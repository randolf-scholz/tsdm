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

from tsdm.types.aliases import Axis


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
    """

    # (..., $N, *d), (..., $N, *d) -> (...)
    def __call__(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute a loss between the predictions and the targets.

        .. signature:: ``[(..., *t, 𝐧), (..., *t, 𝐧)] -> 0``

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
    """

    # (..., $N, *d), (..., $N, *d), (..., $N) -> (...)
    def __call__(
        self, *, predictions: Tensor, targets: Tensor, time_steps: Tensor
    ) -> Tensor: ...


class BaseMetric(nn.Module, Metric):
    r"""Base class for a sample-wise loss function."""

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


class SequentialBaseMetric(BaseMetric):
    r"""Base class for a time-series function.

    Because the loss is computed over a sequence of variable length, the default is to normalize
    the loss by the sequence length, so that loss values are comparable across sequences.
    This class can be used to express decomposable losses of the form

    .. math:: 𝓛(x，x̂) ≔ 𝐀_t ℓ(x_t，x̂_t)

    By default, the aggregation $𝐀_t$ is the mean over the time-axes $𝐄_t$, but simply
    summing over the time-axes is also possible.
    """

    # Constants
    time_axis: Final[int]
    r"""CONST: The time-axes over which the loss is computed."""
    channel_axes: Final[tuple[int, ...]]
    r"""CONST: The channel-axes over which the loss is computed."""
    combined_axes: Final[tuple[int, ...]]
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
        weight: Tensor | None = None,
        axis: Axis = -1,
        time_axis: int = -2,
        normalize_time: bool = True,
        normalize: bool = False,
        learnable: bool = False,
    ) -> None:
        super().__init__(
            axis=axis,
            normalize=normalize,
            weight=weight,
            learnable=learnable,
        )
        self.channel_axes = self.axis  # alias
        self.normalize_channels = self.normalize  # alias

        self.normalize_time = bool(normalize_time)
        self.time_axis = int(time_axis)
        self.combined_axes = (self.time_axis, *self.channel_axes)

        if not {self.time_axis}.isdisjoint(self.channel_axes):
            raise ValueError("Time and channel axes must be disjoint!")

    @abstractmethod
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError
