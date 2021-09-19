r"""Implementation  / loading mechanism for models."""

import logging
from typing import Any, Final

from tsdm.models.models import BaseModel
from tsdm.models.ode_rnn import ODE_RNN

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "Model",
    "MODELS",
    "BaseModel",
    "ODE_RNN",
]


Model = Any
r"""Type hint for models."""

# TODO: replace Any with BaseModel class
MODELS: Final[dict[str, Any]] = {
    "ODE_RNN": ODE_RNN,
}
r"""Dictionary containing all available models."""
