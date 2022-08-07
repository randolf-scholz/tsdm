r"""Linear Algebra Subroutines."""

__all__ = [
    # Modules
    "regularity_tests",
    # Functions
    "erank",
    "col_corr",
    "row_corr",
    "closest_symm",
    "closest_skew",
    "closest_orth",
    "closest_diag",
    "reldist",
    "reldist_diag",
    "reldist_symm",
    "reldist_skew",
    "reldist_orth",
    "relative_error",
    "scaled_norm",
    "grad_norm",
    "multi_scaled_norm",
    "multi_norm",
    "aggregate_and",
    "aggregate_or",
    "cumulative_and",
    "cumulative_or",
]

from tsdm.linalg import regularity_tests
from tsdm.linalg._logical_operators import (
    aggregate_and,
    aggregate_or,
    cumulative_and,
    cumulative_or,
)
from tsdm.linalg._matrix_functions import (
    closest_diag,
    closest_orth,
    closest_skew,
    closest_symm,
    col_corr,
    erank,
    reldist,
    reldist_diag,
    reldist_orth,
    reldist_skew,
    reldist_symm,
    row_corr,
)
from tsdm.linalg._norms import (
    grad_norm,
    multi_norm,
    multi_scaled_norm,
    relative_error,
    scaled_norm,
)
