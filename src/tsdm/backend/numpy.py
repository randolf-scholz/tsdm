r"""Implements `numpy`-backend for tsdm."""

__all__ = [
    "scalar",
    "drop_null",
    "copy_like",
    "apply_along_axes",
]

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Any

from tsdm.types.aliases import Axis


def scalar(x: Any, /, dtype: Any) -> Any:
    r"""Construct a scalar (0d)-array of a given type."""
    array = np.array(x).astype(dtype).squeeze()
    if array.shape != ():
        raise ValueError(f"Created scalar array has shape {array.shape}!")
    # NOTE: we use flatten()[0] instead of item(),
    #   because the latter returns a python scalar, which is not compatible at times.
    return array


def drop_null(x: NDArray, /) -> NDArray:
    r"""Drop `NaN` values from an array, flattening it."""
    return x[~np.isnan(x)].flatten()


def copy_like(x: ArrayLike, ref: NDArray, /) -> NDArray:
    r"""Create an array of the same shape as `x`."""
    return np.array(x, dtype=ref.dtype, copy=False)


def apply_along_axes(
    op: Callable[..., NDArray],
    /,
    *arrays: NDArray,
    axis: Axis,
) -> NDArray:
    r"""Apply a function to multiple arrays along axes.

    It is assumed that all arrays have the same shape.
    """
    if len(arrays) < 1:
        raise ValueError("At least one array is required!")
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("All arrays must have the same shape!")

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
