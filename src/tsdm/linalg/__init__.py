r"""Linear Algebra Subroutines."""

__all__ = [
    # Logical Operators
    "cumulative_and",
    "cumulative_or",
    # Matrix Functions
    "closest_diagonal",
    "closest_orthogonal",
    "closest_skew",
    "closest_symmetric",
    "col_corr",
    "erank",
    "logarithmic_norm",
    "matrix_norm",
    "operator_norm",
    "reldist",
    "reldist_diagonal",
    "reldist_orthogonal",
    "reldist_skew",
    "reldist_symmetric",
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
    "norm",
    "relative_error",
    "scaled_norm",
    "tensor_norm",
]

from tsdm.linalg._logical_operators import (
    cumulative_and,
    cumulative_or,
)
from tsdm.linalg._matrix_functions import (
    closest_diagonal,
    closest_orthogonal,
    closest_skew,
    closest_symmetric,
    col_corr,
    erank,
    logarithmic_norm,
    matrix_norm,
    operator_norm,
    reldist,
    reldist_diagonal,
    reldist_orthogonal,
    reldist_skew,
    reldist_symmetric,
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
    norm,
    relative_error,
    scaled_norm,
    tensor_norm,
)
