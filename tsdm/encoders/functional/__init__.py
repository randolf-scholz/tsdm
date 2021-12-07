r"""Implementation of encoders.

Notes
-----
Contains encoders in functional form.
  - See :mod:`tsdm.encoders.modular` for modular implementations.
"""

__all__ = [
    # Constants
    "FunctionalEncoder",
    "FunctionalEncoders",
    # Functions
    "make_dense_triplets",
    "make_masked_format",
    "make_sparse_triplets",
    "time2float",
    "time2int",
    "triplet2dense",
    "timefeatures",
    # Functions from sklearn
    "binarize",
    "label_binarize",
    "maxabs_scale",
    "minmax_scale",
    "normalize",
    "power_transform",
    "quantile_transform",
    "robust_scale",
    "scale",
]

import logging
from typing import Callable, Final

from sklearn.preprocessing import (
    binarize,
    label_binarize,
    maxabs_scale,
    minmax_scale,
    normalize,
    power_transform,
    quantile_transform,
    robust_scale,
    scale,
)

from tsdm.encoders.functional._functional import (
    make_dense_triplets,
    make_masked_format,
    make_sparse_triplets,
    time2float,
    time2int,
    timefeatures,
    triplet2dense,
)
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

FunctionalEncoder = Callable
r"""Type hint for functional encoders."""

SklearnFunctionalEncoders: Final[LookupTable[FunctionalEncoder]] = {
    "binarize": binarize,
    "label_binarize": label_binarize,
    "maxabs_scale": maxabs_scale,
    "minmax_scale": minmax_scale,
    "normalize": normalize,
    "power_transform": power_transform,
    "quantile_transform": quantile_transform,
    "robust_scale": robust_scale,
    "scale": scale,
}

FunctionalEncoders: Final[LookupTable[FunctionalEncoder]] = {
    "make_dense_triplets": make_dense_triplets,
    "make_masked_format": make_masked_format,
    "make_sparse_triplets": make_sparse_triplets,
    "time2float": time2float,
    "time2int": time2int,
    # "triplet2dense": triplet2dense,
} | SklearnFunctionalEncoders
r"""Dictionary of all available functional encoders."""
