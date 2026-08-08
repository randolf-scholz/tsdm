r"""Linear Algebra Subroutines."""

__all__ = [
    # Logical Operators
    "cumulative_and",
    "cumulative_or",
    "cumulative_xor",
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
    "scaled_norm",
    "tensor_norm",
    # rnn
    "collate_packed",
    "collate_padded",
    "unpad_sequence",
    "unpack_sequence",
]

from ._logical_operators import cumulative_and, cumulative_or, cumulative_xor
from ._matrix_functions import (
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
from ._tensor_functions import (
    geometric_mean,
    grad_norm,
    multi_norm,
    norm,
    scaled_norm,
    tensor_norm,
)
from .collate import collate_packed, collate_padded, unpack_sequence, unpad_sequence
