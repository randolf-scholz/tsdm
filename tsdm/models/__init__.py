r"""Implementation  / loading mechanism for models."""

import logging
from typing import Any, Final

from tsdm.models.model import BaseModel
from tsdm.models.ode_rnn import ODE_RNN

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "BaseModel",
    "ODE_RNN",
    "MODELS",
]

# TODO: replace Any with BaseModel class
MODELS: Final[dict[str, Any]] = {
    "ODE_RNN": ODE_RNN,
}
