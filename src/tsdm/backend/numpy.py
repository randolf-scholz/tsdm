"""Implements `numpy`-backend for tsdm."""

__all__ = [
    "numpy_like",
    "numpy_apply_along_axes",
]

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsdm.types.aliases import Axis


def numpy_like(x: ArrayLike, ref: NDArray, /) -> NDArray:
    r"""Create an array of the same shape as `x`."""
    return np.array(x, dtype=ref.dtype, copy=False)


def numpy_apply_along_axes(
    op: Callable[..., NDArray],
    /,
    *arrays: NDArray,
    axis: Axis,
) -> NDArray:
    r"""Apply a function to multiple arrays along axes.

    It is assumed that all arrays have the same shape.
    """
    assert len({a.shape for a in arrays}) <= 1, "all arrays must have the same shape"
    assert len(arrays) >= 1, "at least one array is required"

    match axis:
        case None:
            axes: tuple[int, ...] = ()
        case int(ax):
            axes = (ax,)
        case _:
            axes = tuple(axis)

    rank = len(arrays[0].shape)
    source = tuple(range(rank))
    inverse_permutation = axes + tuple(ax for ax in range(rank) if ax not in axes)
    perm = tuple(np.argsort(inverse_permutation))
    arrays = tuple(np.moveaxis(array, source, perm) for array in arrays)
    result = op(*arrays)
    result = np.moveaxis(result, source, inverse_permutation)
    return result
