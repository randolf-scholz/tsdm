r"""Generic kernels that work for a variety of backends."""

__all__ = [
    # kernels
    "false_like",
    "true_like",
    "where",
    # extras
    "floor",
    "get_broadcast",
    "get_outlier_options",
    "is_nan",
    "is_singleton",
    "ndim",
    "numel",
    "round",
    "round_impl",
]

from collections.abc import Mapping
from math import prod
from typing import Any, cast

from tsdm.types import Axis, HasShape, SupportsEquality

type FloatArray = Any
type BooleanArray = Any


def ndim(x: HasShape, /) -> int:
    r"""Return the number of dimensions of an object with a `shape` attribute."""
    return len(x.shape)


def numel(x: HasShape, /) -> int:
    r"""Return the number of elements of an object with a `shape` attribute."""
    return prod(x.shape)


def is_singleton(x: HasShape, /) -> bool:
    r"""Determine whether an object with a `shape` attribute has one element."""
    return prod(x.shape) == 1


# FIXME: https://github.com/python/typing/issues/548
def is_nan(x: SupportsEquality, /) -> BooleanArray:
    r"""Determines whether an element is NaN."""
    try:
        return x.isnan()  # type: ignore
    except AttributeError:
        return x != x


# FIXME: https://github.com/python/typing/issues/548
def false_like(x: SupportsEquality, /) -> BooleanArray:
    r"""Return an all-false boolean array with the same shape as `x`."""
    z = x == x
    return z ^ z


# FIXME: https://github.com/python/typing/issues/548
def true_like(x: SupportsEquality, /) -> BooleanArray:
    r"""Return an all-true boolean array with the same shape as `x`."""
    # NOTE: cannot use ~false_like(x) because for float types:
    #   ``(𝙽𝚊𝙽 == 𝙽𝚊𝙽) == False and (𝙽𝚊𝙽 != 𝙽𝚊𝙽) == True``
    z = x == x
    return z ^ (~z)


# FIXME: https://github.com/python/typing/issues/548
def where[Arr: FloatArray](mask: Any, x: Arr, y: Arr, /) -> Arr:
    r"""Select elements from `x` where `mask` is true and from `y` otherwise."""
    # Note: A fundamental problem with implementing this generically is dealing
    #   with infinity and NaN values.
    #   We use a hack to avoid this problem: by IEEE standard x⁰ = 1,
    #   even when x ∈ {±∞. NaN}.
    #   therefore, what we can do is use xᵐ = {x: m=1, 1: x=0} to select between x and 1.
    #   in particular, xᵐ - (1-m) = {x: m=1, 0: m=0}
    #   so, we can return (xᵐ - (1-m)) + (y¹⁻ᵐ - m) = xᵐ + y¹⁻ᵐ - 1
    m = cast("Arr", mask * 1.0)  # floating conversion
    return x**m + y ** (1.0 - m) - 1.0


def floor[Arr: FloatArray](x: Arr, /) -> Arr:
    r"""Floor elements of the array to the nearest integer less than or equal to x."""
    # Note: x // 1 produces NaN when x is ±inf, instead we use x - (x % 1)
    r = x % 1  # fractional part, NaN for ±inf
    m = is_nan(r)
    return where(m, x, x - r)


def round_impl[Arr: FloatArray](x: Arr, /, *, decimals: int = 0) -> Arr:
    r"""Round elements of the array to the given number of decimals."""
    # shift the decimal point to the right
    factor = 10**decimals
    x = x * factor

    # round half-to-even
    z = x + 0.5
    r = floor(z)  # r = ⌊x + ½⌋

    # fix incorrect rounding when z%2 is exactly equal to 1.0
    y = where((z % 2) == 1.0, r - 1.0, r)

    # shift the decimal point back
    y = y / factor

    return y


def round[Arr: FloatArray](x: Arr, /, *, decimals: int = 0) -> Arr:  # ruff: ignore[A001]
    r"""Round elements of the array to the given number of decimals.

    Rounding uses [round half to even](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even).

    Note:
        Rounding does not explicitly cast the result to an integer dtype.

    Args:
        x: Input array.
        decimals: Number of decimal places to round to. Defaults to zero.

    Returns:
        Array with elements rounded to the given number of decimals.
    """
    try:
        return x.round(decimals=decimals)
    except AttributeError:
        return round_impl(x, decimals=decimals)


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
