r"""Implementations of activation functions.

Notes
-----
Contains activations in both functional and modular form.
  - See :mod:`tsdm.util.activations.functional` for functional implementations.
  - See :mod:`tsdm.util.activations.modular` for modular implementations.
"""
from __future__ import annotations

__all__ = [
    # Sub-Modules
    "functional",
    "modular",
    # Constants
    "Activation",
    "ACTIVATIONS",
    "ModularActivations",
    "ModularActivation",
    "ModularActivationType",
    "FunctionalActivation",
    "FunctionalActivations",
    "FunctionalActivationType"
]

import logging
from typing import Final, Union

from tsdm.util.activations import functional, modular
from tsdm.util.activations.functional import FunctionalActivation, FunctionalActivations,FunctionalActivationType
from tsdm.util.activations.modular import ModularActivation, ModularActivations, ModularActivationType

LOGGER = logging.getLogger(__name__)

Activation = Union[FunctionalActivation, ModularActivation]
r"""Type hint for activations."""
ActivationType = Union[FunctionalActivationType, ModularActivationType]
r"""Type hint for activations."""
ACTIVATIONS: Final[dict[str, ActivationType]] = {
    **ModularActivations,
    **FunctionalActivations,
}
r"""Dictionary of all available activations."""
