r"""Statistical Analysis."""

__all__ = [
    # Functions
    "approx_float_gcd",
    "data_overview",
    "float_gcd",
    "is_quasiregular",
    "is_regular",
    "regularity_coefficient",
    "time_gcd",
]

from ._stats import data_overview
from .regularity_tests import (
    approx_float_gcd,
    float_gcd,
    is_quasiregular,
    is_regular,
    regularity_coefficient,
    time_gcd,
)
