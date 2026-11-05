r"""Utility functions for linear algebra operations."""

__all__ = [
    "get_broadcast",
    "reduce_axes",
    "invert_axis_selection",
]

from collections.abc import Iterable
from types import EllipsisType
from typing import overload

from tsdm.types.aliases import Axis, IndexArgND


def get_broadcast(
    original_shape: tuple[int, ...],
    /,
    *,
    axis: Axis,
    keepdim: bool = False,
) -> tuple[slice | None, ...]:
    r"""Create an indexer that restores axes removed by a tensor contraction.

    If `x = contraction(data, axis=axis)`, then `x[get_broadcast(data.shape, axis=axis)]`
    has the same shape as `contraction(data, axis=axis, keepdims=True)`.
    This allows element-wise operations such as `data + x[broadcast]`.

    With `keepdim=True`, `axis` instead selects the axes retained by the
    contraction. Negative axes are normalized modulo the rank.

    Args:
        original_shape: The shape of the tensor before contraction.
        axis: The axes to contract, or `None` to select all axes.
        keepdim: Whether `axis` selects retained axes instead of contracted axes.

    Example:
        ```pycon
        >>> import numpy as np
        >>> arr = np.ones([1, 2, 3, 4, 5])
        >>> ax = (1, -1)
        >>> broadcast = get_broadcast(arr.shape, axis=ax)
        >>> m = np.mean(arr, axis=ax)
        >>> m_ref = np.mean(arr, axis=ax, keepdims=True)
        >>> m[broadcast].shape == m_ref.shape
        True
        ```

        For shape `(2, 3, 4, 5, 6, 7)` and contracted axes `(0, 2, -1)`,
        the indexer is `(None, slice(None), None, slice(None), slice(None), None)`.
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
    r"""Reduce an axis selection to account for tensor indexing.

    Given an operation `op(T, axis=axis)`, use
    `reduced_axis = reduce_axes(axis, selection)` to obtain the axis selection
    for `op(T[selection], axis=reduced_axis)`.

    An `axis` of `None` remains `None`. Inserting dimensions with `None` is
    unsupported for explicit axis selections; at most one ellipsis is allowed.
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


def invert_axis_selection(axis: Axis, /, *, ndim: int) -> tuple[int, ...]:
    r"""Return the complement of an axis selection for a tensor of rank `ndim`.

    Negative axes are normalized modulo `ndim`. An `axis` of `None` selects
    every axis, so its complement is empty. The output order is unspecified.

    The following examples show the complementary axes in sorted order:

    | `ndim` | `axis`         | Complement     |
    |--------|----------------|----------------|
    | 4      | `None`         | `()`           |
    | 4      | `(-3, -2, -1)` | `(0,)`         |
    | 4      | `(-2, -1)`     | `(0, 1)`       |
    | 4      | `(-1,)`        | `(0, 1, 2)`    |
    | 4      | `()`           | `(0, 1, 2, 3)` |
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
