r"""Implementation  / loading mechanism for models.

TODO: Module Summary.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "Model",
    "MODELS",
    # Classes
    "BaseModel",
    "ODE_RNN",
]


import logging
from typing import Final

from torch import nn

from tsdm.models.models import BaseModel
from tsdm.models.ode_rnn import ODE_RNN

LOGGER = logging.getLogger(__name__)


Model = nn.Module
r"""Type hint for models."""

# TODO: replace Any with BaseModel class
MODELS: Final[dict[str, type[Model]]] = {
    "ODE_RNN": ODE_RNN,
}
r"""Dictionary containing all available models."""
