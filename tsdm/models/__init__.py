r"""Implementation  / loading mechanism for models.

tsdm.models
===========
"""

from .model import BaseModel
from .ode_rnn import ODE_RNN

__all__ = ["BaseModel", "ODE_RNN"]
