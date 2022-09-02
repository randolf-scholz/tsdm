r"""Linear Algebra Subroutines."""

__all__ = [
    # Functions
    "aggregate_and",
    "aggregate_or",
    "closest_diag",
    "closest_orth",
    "closest_skew",
    "closest_symm",
    "col_corr",
    "cumulative_and",
    "cumulative_or",
    "erank",
    "grad_norm",
    "multi_norm",
    "multi_scaled_norm",
    "relative_error",
    "reldist",
    "reldist_diag",
    "reldist_orth",
    "reldist_skew",
    "reldist_symm",
    "row_corr",
    "scaled_norm",
]

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
