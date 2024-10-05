r"""Generic kernels that work for a variety of backends."""

__all__ = [
    # Functions
    "false_like",
    "generic_floor",
    "generic_ndim",
    "generic_round",
    "generic_where",
    "generic_numel",
    "is_nan",
    "is_singleton",
    "round",
    "true_like",
]

from math import prod
from typing import Any, cast

from tsdm.types.arrays import NumericalArray, SupportsShape
from tsdm.types.scalars import FloatScalar


def generic_ndim(x: SupportsShape, /) -> int:
    r"""Returns the number of dimensions in the tensor like object."""
    return len(x.shape)


def generic_numel(x: SupportsShape, /) -> int:
    r"""Returns the number of elements in the tensor like object."""
    return prod(x.shape)


def is_singleton(x: SupportsShape, /) -> bool:
    r"""Determines whether a tensor like object has a single element."""
    return prod(x.shape) == 1


# FIXME: improve type hints (HKTs)
def is_nan(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Determines whether an element is NaN."""
    try:
        return x.isnan()  # type: ignore[attr-defined]
    except AttributeError:
        return x != x


# FIXME: improve type hints (HKTs)
def false_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    z = x == x
    return z ^ z


# FIXME: improve type hints (HKTs)
def true_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    # NOTE: cannot use ~false_like(x) because for float types:
    #   `(𝙽𝚊𝙽 == 𝙽𝚊𝙽) == False and (𝙽𝚊𝙽 != 𝙽𝚊𝙽) == True`
    z = x == x
    return z ^ (~z)


# FIXME: improve type hints (HKTs)
def generic_where[Arr: NumericalArray](mask: Any, x: Arr, y: Arr, /) -> Arr:
    r"""Selects elements from `x` or `y` based on `m`."""
    # Note: A fundamental problem with implementing this generically is dealing
    #   with infinity and NaN values.
    #   We use a hack to avoid this problem: by IEEE standard x⁰ = 1,
    #   even when x ∈ {±∞. NaN}.
    #   therefore, what we can do is use xᵐ = {x: m=1, 1: x=0} to select between x and 1.
    #   in particular, xᵐ - (1-m) = {x: m=1, 0: m=0}
    #   so, we can return (xᵐ - (1-m)) + (y¹⁻ᵐ - m) = xᵐ + y¹⁻ᵐ - 1
    m = cast(Arr, mask * 1.0)  # floating conversion
    return x**m + y ** (1.0 - m) - 1.0


def generic_floor[Arr: NumericalArray[FloatScalar]](x: Arr, /) -> Arr:
    r"""Floor elements of the array to the nearest integer less than or equal to x."""
    # Note: x // 1 produces NaN when x is ±inf, instead we use x - (x % 1)
    r = x % 1  # fractional part, NaN for ±inf
    m = is_nan(r)
    return generic_where(m, x, x - r)


def generic_round[Arr: NumericalArray[FloatScalar]](
    x: Arr, /, *, decimals: int = 0
) -> Arr:
    r"""Round elements of the array to the given number of decimals."""
    # shift the decimal point to the right
    factor = 10**decimals
    x = x * factor

    # round half-to-even
    z = x + 0.5
    r = generic_floor(z)  # r = ⌊x + ½⌋

    # fix incorrect rounding when z%2 is exactly equal to 1.0
    y = generic_where((z % 2) == 1.0, r - 1.0, r)

    # shift the decimal point back
    y = y / factor

    return y


def round[Arr: NumericalArray[FloatScalar]](x: Arr, /, *, decimals: int = 0) -> Arr:  # noqa: A001
    r"""Round elements of the array to the given number of decimals.

    Note:
        Keeps floating point data type, does not cast to integer.

    Args:
        x: Input array
        decimals: Number of decimal places to round to (default: 0)

    Returns:
        Array with elements rounded to the given number of decimals

    Refernces:
        https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
    """
    try:
        return x.round(decimals=decimals)  # type: ignore[attr-defined]
    except AttributeError:
        return generic_round(x, decimals=decimals)
