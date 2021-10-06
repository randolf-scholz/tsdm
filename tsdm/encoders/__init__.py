r"""Implementation of encoders.

Notes
-----
Contains encoders in both modular and functional form.
  - See :mod:`tsdm.encoders.functional` for functional implementations.
  - See :mod:`tsdm.encoders.modular` for modular implementations.
"""
#  TODO:
# - Target Encoding: enc(x) = mean(enc(y|x))
# - Binary Encoding: enx(x) =
# - Hash Encoder: enc(x) = binary(hash(x))
# - Effect/Sum/Deviation Encoding:
# - Sum Encoding
# - ECC Binary Encoding:
# - Ordinal Coding: (cᵢ | i=1:n) -> (i| i=1...n)
# - Dummy Encoding: like one-hot, with (0,...,0) added as a category
# - word2vec
# - Learned encoding:
#
# Hierarchical Categoricals:
# - Sum Coding
# - Helmert Coding
# - Polynomial Coding
# - Backward Difference Coding:

from __future__ import annotations

__all__ = [
    # Constants
    "Encoder",
    "EncoderType",
    "ENCODERS",
    "ModularEncoder",
    "ModularEncoders",
    "ModularEncoderType",
    "FunctionalEncoder",
    "FunctionalEncoders",
    "FunctionalEncoderType",
]

import logging
from typing import Final, Union

from tsdm.encoders import functional, modular
from tsdm.encoders.functional import (
    FunctionalEncoder,
    FunctionalEncoders,
    FunctionalEncoderType,
)
from tsdm.encoders.modular import ModularEncoder, ModularEncoders, ModularEncoderType

LOGGER = logging.getLogger(__name__)

Encoder = Union[ModularEncoder, FunctionalEncoder]
r"""Type hint for encoders."""
EncoderType = Union[ModularEncoderType, FunctionalEncoderType]
r"""Type hint for encoders."""
ENCODERS: Final[dict[str, EncoderType]] = {**ModularEncoders, **FunctionalEncoders}
r"""Dictionary of all available encoders."""
