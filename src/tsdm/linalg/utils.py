r"""Utility functions for linear algebra operations."""

__all__ = [
    "get_broadcast",
    "reduce_axes",
    "invert_axis_selection",
]

from collections.abc import Iterable
from types import EllipsisType
from typing import Optional, overload

from tsdm.types.aliases import Axis, IndexArgND


def get_broadcast(
    original_shape: tuple[int, ...],
    /,
    *,
    axis: Axis,
    keepdim: bool = False,
) -> tuple[slice | None, ...]:
    r"""Creates an indexer that broadcasts a tensors contracted via `axis`.

    Essentially works like a-posteriori adding the `keepdims=True` option to a contraction.
    If `x = contraction(data, axis)` (e.g. sum, mean, max), then ``x[broadcast]``
    for ``broadcast=get_braodcast(data.shape, axis) is roughly equivalent to
    ``contraction(data, axis, keepdims=True)``.

    This achieves element-wise compatibility: ``data + x[broadcast]``.

    >>> import numpy as np
    >>> arr = np.random.randn(1, 2, 3, 4, 5)
    >>> ax = (1, -1)
    >>> broadcast = get_broadcast(arr.shape, axis=ax)
    >>> m = np.mean(arr, ax)
    >>> m_ref = np.mean(arr, axis=ax, keepdims=True)
    >>> m[broadcast].shape == m_ref.shape
    True

    If `keepdim` is True, then the broadcast is the complement of the contraction,
    i.e. ``x[broadcast]`` is roughly equivalent to ``contraction(data, kept_axis, keepdims=True)``,
    where ``kept_axis = set(range(data.ndim)) - set(ax%data.ndim for ax in axis)``.

    Args:
        original_shape: The tensor to be contracted.
        axis: The axes to be contracted.
        keepdim: select `True` if the axes are to be kept instead.

    Example:
        data is of shape  ``(2,3,4,5,6,7)``
        axis is the tuple ``(0,2,-1)``
        broadcast is ``(:, None, :, None, None, :)``
    """
    rank = len(original_shape)

    if keepdim:
        match axis:
            case None:  # all axes are contracted
                kept_axis = set()
            case int():
                kept_axis = {axis % rank}
            case _:
                kept_axis = {a % rank for a in axis}
        return tuple(slice(None) if a in kept_axis else None for a in range(rank))

    match axis:
        case None:  # all axes are contracted
            contracted_axes = set(range(rank))
        case int():
            contracted_axes = {axis % rank}
        case _:
            contracted_axes = {a % rank for a in axis}

    return tuple(None if a in contracted_axes else slice(None) for a in range(rank))


@overload
def reduce_axes(axis: None, selection: IndexArgND) -> None: ...
@overload
def reduce_axes(
    axis: int | tuple[int, ...], selection: str | list[str] | IndexArgND
) -> tuple[int, ...]: ...
def reduce_axes(
    axis: Axis, selection: str | list[str] | IndexArgND
) -> tuple[int, ...] | None:
    r"""Returns axis selection corresponding to given tensor indexing.

    Assuming some universal operator `op` acts in tensor `T`, that is `op(T, axis=axis)`,
    then the return of this method allows to apply `op(T[selection], axis=reduced_axis)`.
    """
    # convert to tuple
    match axis:
        case None:
            return None
        case []:
            return ()
        case int(a):
            axis = (a,)
        case _:
            axis = tuple(axis)

    match selection:
        case None:
            raise NotImplementedError("Slicing with None not implemented.")
        case int() | str():
            return axis[1:]
        case EllipsisType():
            return axis
        case list(seq):
            drop = len(seq) <= 1
            return axis[drop:]
        case slice() as slc:
            drop = _slice_size(slc) in {0, 1}
            return axis[drop:]
        case range(start=start, stop=stop):
            drop = (stop - start) in {0, 1}
            return axis[drop:]
        case tuple(tup):
            if sum(x is Ellipsis for x in tup) > 1:
                raise ValueError("Only one Ellipsis is allowed.")
            if len(tup) == 0:
                return axis
            if Ellipsis in tup:
                idx = tup.index(Ellipsis)
                return (
                    reduce_axes(axis[:idx], tup[:idx])
                    + reduce_axes(axis[idx : idx + len(tup) - 1], tup[idx])
                    + reduce_axes(axis[idx + len(tup) - 1 :], tup[idx + 1 :])
                )
            # recurse on the first element
            return reduce_axes(axis[:1], tup[0]) + reduce_axes(axis[1:], tup[1:])
        case _:
            raise TypeError(f"Unknown type {type(selection)}")


def _slice_size(s: slice, /) -> Optional[int]:
    r"""Get the size of a slice."""
    if s.stop is None or s.start is None:
        return None
    return s.stop - s.start


def invert_axis_selection(axis: Axis, /, *, ndim: int) -> tuple[int, ...]:
    r"""Invert axes-selection for a rank `ndim` tensor.

    Example:
        +------+------------+-----------+
        | ndim | axis       | inverted  |
        +======+============+===========+
        | 4    | None       | ()        |
        +------+------------+-----------+
        | 4    | (-3,-2,-1) | (0,)      |
        +------+------------+-----------+
        | 4    | (-2,-1)    | (0,1)     |
        +------+------------+-----------+
        | 4    | (-1)       | (0,1,2)   |
        +------+------------+-----------+
        | 4    | ()         | (0,1,2,3) |
        +------+------------+-----------+
    """
    match axis:
        case None:
            return ()
        case int():
            return tuple(set(range(ndim)) - {axis % ndim})
        case Iterable():
            return tuple(set(range(ndim)) - {a % ndim for a in axis})
        case _:
            raise TypeError(f"axis must be None, int, or Iterable, not {type(axis)}")
