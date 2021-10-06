r"""Functional variants of encoders."""

from __future__ import annotations

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
    # Functions
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
from typing import Any, Final

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

from tsdm.encoders.functional._encoders import (
    make_dense_triplets,
    make_masked_format,
    make_sparse_triplets,
    time2float,
    time2int,
    triplet2dense,
)

LOGGER = logging.getLogger(__name__)

FunctionalEncoder = Any
r"""Type hint for encoders."""


SklearnFunctionalEncoders: Final[dict[str, FunctionalEncoder]] = {
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

FunctionalEncoders: Final[dict[str, FunctionalEncoder]] = {
    "make_dense_triplets": make_dense_triplets,
    "make_masked_format": make_masked_format,
    "make_sparse_triplets": make_sparse_triplets,
    "time2float": time2float,
    "time2int": time2int,
    "triplet2dense": triplet2dense,
} | SklearnFunctionalEncoders
r"""Dictionary of all available encoders."""
