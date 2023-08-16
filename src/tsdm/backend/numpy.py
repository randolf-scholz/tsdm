"""implement pandas backend for tsdm."""


__all__ = [
    "numpy_like",
]

import numpy as np
from numpy.typing import ArrayLike, NDArray


def numpy_like(x: ArrayLike, ref: NDArray, /) -> NDArray:
    """Create an array of the same shape as `x`."""
    return np.array(x, dtype=ref.dtype, copy=False)
