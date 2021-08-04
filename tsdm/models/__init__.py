r"""Implementation  / loading mechanism for models.

tsdm.models
===========
"""

from tsdm.models.model import BaseModel
from tsdm.models.ode_rnn import ODE_RNN


__all__ = [
    "BaseModel",
    "ODE_RNN",
    "MODELS",
]

MODELS = {"ODE_RNN": ODE_RNN}
