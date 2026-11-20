r"""Generic kernels that work for a variety of backends."""

__all__ = [
    # Functions
    "false_like",
    "floor",
    "round_impl",
    "is_nan",
    "is_singleton",
    "ndim",
    "numel",
    "round",
    "true_like",
    "where",
]

from math import prod
from typing import Any, cast

from tsdm.types import HasShape, SupportsEquality

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
