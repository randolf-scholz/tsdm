r"""Implementations of activation functions.

Notes
-----
Contains activations in both functional and modular form.
  - See :mod:`tsdm.util.activations.functional` for functional implementations.
  - See :mod:`tsdm.util.activations.modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modular",
    # Constants
    "Activation",
    "ACTIVATIONS",
    "ModularActivations",
    "ModularActivation",
    "FunctionalActivation",
    "FunctionalActivations",
]

import logging
from typing import Final, Union

from tsdm.models.activations import functional, modular
from tsdm.models.activations.functional import (
    FunctionalActivation,
    FunctionalActivations,
)
from tsdm.models.activations.modular import ModularActivation, ModularActivations
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

Activation = Union[FunctionalActivation, ModularActivation]
r"""Type hint for activations."""

ACTIVATIONS: Final[
    LookupTable[Union[FunctionalActivation, type[ModularActivation]]]
] = {
    **ModularActivations,
    **FunctionalActivations,
}
r"""Dictionary of all available activations."""
