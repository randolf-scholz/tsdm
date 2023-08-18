r"""Linear Algebra Subroutines."""

__all__ = [
    # Logical Operators
    "aggregate_and",
    "aggregate_or",
    "cumulative_and",
    "cumulative_or",
    # Matrix Functions
    "closest_diag",
    "closest_orth",
    "closest_skew",
    "closest_symm",
    "col_corr",
    "erank",
    "logarithmic_norm",
    "matrix_norm",
    "operator_norm",
    "reldist",
    "reldist_diag",
    "reldist_orth",
    "reldist_skew",
    "reldist_symm",
    "relerank",
    "row_corr",
    "schatten_norm",
    "spectral_abscissa",
    "spectral_radius",
    "stiffness_ratio",
    # Tensor Functions
    "geometric_mean",
    "grad_norm",
    "multi_norm",
    "tensor_norm",
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
    logarithmic_norm,
    matrix_norm,
    operator_norm,
    reldist,
    reldist_diag,
    reldist_orth,
    reldist_skew,
    reldist_symm,
    relerank,
    row_corr,
    schatten_norm,
    spectral_abscissa,
    spectral_radius,
    stiffness_ratio,
)
from tsdm.linalg._tensor_functions import (
    geometric_mean,
    grad_norm,
    multi_norm,
    tensor_norm,
)
