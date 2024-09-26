r"""Generic kernels that work for a variety of backends."""

__all__ = [
    # Functions
    "false_like",
    "is_scalar",
    "is_singleton",
    "true_like",
    "round",
]

from math import prod

import pyarrow as pa

from tsdm.types.protocols import NumericalArray, SupportsRound, SupportsShape


def false_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    z = x == x
    return z ^ z


def true_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    # NOTE: cannot use ~false_like(x) because for float types:
    #   `(𝙽𝚊𝙽 == 𝙽𝚊𝙽) == False and (𝙽𝚊𝙽 != 𝙽𝚊𝙽) == True`
    z = x == x
    return z ^ (~z)


def is_singleton(x: SupportsShape, /) -> bool:
    r"""Determines whether a tensor like object has a single element."""
    return prod(x.shape) == 1


def is_scalar(x: object, /) -> bool:
    r"""Determines whether an object is a scalar."""
    return isinstance(x, bool | float | int | pa.Scalar | str)


def round[Arr: NumericalArray](x: Arr, /, *, decimals: int = 0) -> Arr:  # noqa: A001
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
    if isinstance(x, SupportsRound):
        return x.round(decimals=decimals)

    # Now, apply rounding-half-to-even, in accordance with IEEE
    raise NotImplementedError("Rounding half-to-even not implemented.")
