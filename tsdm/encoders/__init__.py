r"""Encoders for different data types.

TODO: Module Summary.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "Encoder",
    "ENCODERS",
    # Functions
    "make_dense_triplets",
    "make_masked_format",
    "make_sparse_triplets",
    "time2float",
    "time2int",
    "triplet2dense",
]

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
