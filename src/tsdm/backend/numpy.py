r"""NumPy backend implementation."""

__all__ = [
    # types
    "NumpyDtype",
    "NumpyDtypeArg",
    # kernels
    "apply_along_axes",
    "copy_like",
    "drop_null",
    "scalar",
    # extras
    "round_relative",
]

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsdm.types.aliases import Axis

type NumpyDtype = np.dtype
r"""Type Alias for `numpy` dtypes."""

type NumpyDtypeArg = str | type | NumpyDtype
r"""Type Alias for `numpy` dtype arguments."""


def scalar(x: Any, /, dtype: Any) -> Any:
    r"""Construct a zero-dimensional array of the given dtype.

    Singleton dimensions are removed. Raise `ValueError` if the resulting array
    is not zero-dimensional.
    """
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
    r"""Create an array from `x` using the dtype of `ref`."""
    return np.array(x, dtype=ref.dtype, copy=False)


def apply_along_axes(
    op: Callable[..., NDArray],
    /,
    *arrays: NDArray,
    axis: Axis,
) -> NDArray:
    r"""Apply a function to multiple arrays along axes.

    Move the selected axes to the front, call `op`, and restore the original axis
    order. The operation must preserve the shape of its inputs.

    Args:
        op: Function applied to the arrays after moving the selected axes.
        *arrays: One or more arrays with identical shapes.
        axis: Axes to move to the front. `None` leaves the axis order unchanged.

    Raises:
        ValueError: If no arrays are supplied or their shapes differ.
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


def round_relative(x: np.ndarray, /, *, decimals: int = 2) -> np.ndarray:
    r"""Round to relative precision.

    For positive values, retain `decimals + 1` significant digits. Zero remains
    zero; negative values produce `NaN` because their logarithm is undefined.

    Examples:
        ```pycon
        >>> with np.printoptions(precision=6, floatmode="fixed", suppress=True):
        ...     round_relative(np.array([1234.5678, 0.12345678]), decimals=2)
        array([1230.000000,    0.123000])

        ```
    """
    order = np.where(x == 0, 0, np.floor(np.log10(x)))
    digits = decimals - order
    rounded = np.rint(x * 10**digits)
    return np.true_divide(rounded, 10**digits)
