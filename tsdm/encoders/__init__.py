r"""Encoders for different data types."""
import logging
from typing import Any, Final

from tsdm.encoders._encoders import (
    make_dense_triplets,
    make_masked_format,
    make_sparse_triplets,
    time2float,
    time2int,
    triplet2dense,
)

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "Encoder",
    "ENCODERS",
    "make_dense_triplets",
    "make_masked_format",
    "make_sparse_triplets",
    "time2float",
    "time2int",
    "triplet2dense",
]

Encoder = Any
r"""Type hint for encoders."""

ENCODERS: Final[dict[str, Encoder]] = {
    "make_dense_triplets": make_dense_triplets,
    "make_masked_format": make_masked_format,
    "make_sparse_triplets": make_sparse_triplets,
    "time2float": time2float,
    "time2int": time2int,
    "triplet2dense": triplet2dense,
}
r"""Dictionary containing all available encoders."""
