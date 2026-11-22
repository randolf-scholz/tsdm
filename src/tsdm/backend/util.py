r"""Utilities shared by backend implementations."""

__all__ = [
    "get_outlier_options",
    "get_broadcast",
]

from collections.abc import Mapping
from typing import Any

from tsdm.types import Axis


def get_outlier_options(
    limits: Mapping[str, Any] | None = None,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = None,
    upper_bound: Mapping[Any, float | None] | float | None = None,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = None,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = None,
) -> dict[str, Any]:
    r"""Validate and normalize outlier bounds from mappings or keyword arguments.

    With `limits`, every boundary keyword must be `None`. Otherwise, missing
    or `None` bounds leave that side unbounded. Explicit `None` keywords may
    accompany `limits`; they are equivalent to omitted keywords.

    Mappings may be organized by bound name or variable name. Native limits
    tables must first be converted to mappings by their backend. Each bounded
    column must have entries in all four boundary mappings.
    """
    lims: dict[str, Any] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }
    if limits is None:
        return lims
    if any(value is not None for value in lims.values()):
        raise ValueError("Limits specified both as positional and keyword arguments.")
    if not isinstance(limits, Mapping):
        raise TypeError(f"Expected a mapping of outlier limits, got {type(limits)}.")
    if lims.keys() <= limits.keys():
        return {key: limits[key] for key in lims}
    return {
        key: {variable: bounds[key] for variable, bounds in limits.items()}
        for key in lims
    }


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
