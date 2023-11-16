"""Universal backend functions that are independent of the backend used."""

__all__ = [
    "false_like",
    "true_like",
    "is_singleton",
    "is_scalar",
    "to_scalar",
]

from math import prod

import numpy as np
import pandas as pd
import pyarrow as pa
from typing_extensions import Any

from tsdm.types.protocols import NumericalArray, SupportsShape


def false_like(x: NumericalArray, /) -> NumericalArray[bool]:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    z = x == x  # pylint: disable=comparison-with-itself
    return z ^ z


def true_like(x: NumericalArray, /) -> NumericalArray[bool]:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    # NOTE: cannot use ~false_like(x) because for float types:
    # (𝙽𝚊𝙽 == 𝙽𝚊𝙽) == False and (𝙽𝚊𝙽 != 𝙽𝚊𝙽) == True
    z = x == x  # pylint: disable=comparison-with-itself
    return z ^ (~z)


def is_singleton(x: SupportsShape, /) -> bool:
    """Determines whether a tensor like object has a single element."""
    return prod(x.shape) == 1
    # numpy: size, len  / shape + prod
    # torch: size + prod / numel / shape + prod
    # table: shape + prod
    # DataFrame: shape + prod
    # Series: shape + prod
    # pyarrow table: shape + prod
    # pyarrow array: ????


def is_scalar(x: Any, /) -> bool:
    """Determines whether an object is a scalar."""
    return (
        isinstance(x, (int, float, str, bool))
        or np.isscalar(x)
        or pd.api.types.is_scalar(x)
        or isinstance(x, pa.Scalar)
    )


def to_scalar():
    """Convert a singleton to a scalar."""
    raise NotImplementedError
