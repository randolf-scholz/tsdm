r"""Fallback backend implementations."""

__all__ = [
    "is_null_scalar",
]

import math

import numpy as np
import pandas as pd
import pyarrow as pa


def is_null_scalar(value: object, /) -> bool:
    r"""Check if a scalar value is considered as NA/Null."""
    return (
        (value is None)
        or (isinstance(value, float) and math.isnan(value))
        or (isinstance(value, np.generic) and np.isnan(value))
        or (value is pd.NA)
        or (value is pd.NaT)
        or (isinstance(value, pa.Scalar) and not value.is_valid)  # pyright: ignore[reportAttributeAccessIssue]
    )
