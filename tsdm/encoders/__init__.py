r"""Implementation of Encoders.

Role & Specification
--------------------

Encoders are used in multiple contexts
  - Perform preprocessing for task objects: For example, a task might ask to evaluate on
    standardized features. In this case, an pre_encoder object is associated with the task that
    will perform this preprocessing at task creation time.
  - Perform data encoding tasks such as encoding of categorical variables.
  - Transform data from one framework to another, like :mod:`numpy` → :mod:`torch`

Specification:
  - Encoders **must** be reversible.
  - Modules that are not reversible, we call transformations.
      - Example: Convert logit output of a NN to a class prediction.

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
    # Sub-Modules
    "functional",
    "modular",
    # Constants
    "Encoder",
    "ENCODERS",
    "ModularEncoder",
    "ModularEncoders",
    "FunctionalEncoder",
    "FunctionalEncoders",
]

import logging
from typing import Final, Union

from tsdm.encoders import functional, modular
from tsdm.encoders.functional import FunctionalEncoder, FunctionalEncoders
from tsdm.encoders.modular import ModularEncoder, ModularEncoders
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

Encoder = Union[FunctionalEncoder, ModularEncoder]
r"""Type hint for encoders."""

ENCODERS: Final[LookupTable[Union[FunctionalEncoder, type[ModularEncoder]]]] = {
    **ModularEncoders,
    **FunctionalEncoders,
}
r"""Dictionary of all available encoders."""
