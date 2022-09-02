r"""Statistical Analysis."""

__all__ = [
    # Sub-Modules
    "regularity_tests",
    # Functions
    "data_overview",
    "sparsity",
    # Regularity tests
    "approx_float_gcd",
    "float_gcd",
    "is_quasiregular",
    "is_regular",
    "regularity_coefficient",
    "time_gcd",
]

from tsdm.random.stats import regularity_tests
from tsdm.random.stats._stats import data_overview, sparsity
from tsdm.random.stats.regularity_tests import (
    approx_float_gcd,
    float_gcd,
    is_quasiregular,
    is_regular,
    regularity_coefficient,
    time_gcd,
)
