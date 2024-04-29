r"""Statistical Analysis."""

__all__ = [
    # Sub-Modules
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
from tsdm.random import distributions
from tsdm.random.stats._stats import data_overview, sparsity
from tsdm.random.stats.regularity_tests import (
    approx_float_gcd,
    float_gcd,
    is_quasiregular,
    is_regular,
    regularity_coefficient,
    time_gcd,
)
