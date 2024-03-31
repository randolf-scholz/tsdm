r"""Implementations of activation functions.

Notes:
    Contains activations in both functional and modular form.
    - See `tsdm.models.activations.functional` for functional implementations.
    - See `tsdm.models.activations.modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "base",
    "functional",
    "modular",
    # Constants
    "ACTIVATIONS",
    "FUNCTIONAL_ACTIVATIONS",
    "MODULAR_ACTIVATIONS",
    "TORCH_ACTIVATIONS",
    "TORCH_FUNCTIONAL_ACTIVATIONS",
    "TORCH_MODULAR_ACTIVATIONS",
    # ABCs & Protocols
    "Activation",
    "ActivationABC",
    # Classes
    "HardBend",
    # Functions
    "geglu",
    "hard_bend",
    "reglu",
    # utils
    "get_activation",
]

from tsdm.models.activations import base, functional, modular
from tsdm.models.activations._torch_imports import (
    TORCH_ACTIVATIONS,
    TORCH_FUNCTIONAL_ACTIVATIONS,
    TORCH_MODULAR_ACTIVATIONS,
)
from tsdm.models.activations.base import Activation, ActivationABC
from tsdm.models.activations.functional import geglu, hard_bend, reglu
from tsdm.models.activations.modular import HardBend

FUNCTIONAL_ACTIVATIONS: dict[str, Activation] = {
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    "reglu": reglu,
    "geglu": geglu,
    "hard_bend": hard_bend,
}
r"""Dictionary containing all available functional activations."""

MODULAR_ACTIVATIONS: dict[str, type[Activation]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    "HardBend": HardBend,
}
r"""Dictionary containing all available activations."""

ACTIVATIONS: dict[str, Activation | type[Activation]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    **MODULAR_ACTIVATIONS,
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    **FUNCTIONAL_ACTIVATIONS,
}
r"""Dictionary containing all available activations."""


def get_activation(activation: object, /) -> Activation:
    r"""Get an activation function by name."""
    match activation:
        case type() as cls:
            return cls()
        case str(name):
            return get_activation(ACTIVATIONS[name])
        case func if callable(func):
            return func
        case _:
            raise TypeError(f"Invalid activation: {activation!r}")
