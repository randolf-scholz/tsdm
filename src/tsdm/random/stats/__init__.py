r"""Statistical Analysis."""

__all__ = [
    # Sub-Modules
    "regularity_tests",
    "distributions",
    # Functions
    "approx_float_gcd",
    "data_overview",
    "float_gcd",
    "is_quasiregular",
    "is_regular",
    "regularity_coefficient",
    "sparsity",
    "time_gcd",
]

from tsdm.random.stats import distributions, regularity_tests
from tsdm.random.stats._stats import data_overview, sparsity
from tsdm.random.stats.regularity_tests import (
    approx_float_gcd,
    float_gcd,
    is_quasiregular,
    is_regular,
    regularity_coefficient,
    time_gcd,
)
