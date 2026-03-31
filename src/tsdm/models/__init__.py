r"""Implementation / loading mechanism for models.

There are two types of models:

- Core models: These consist of only a pytorch/tensorflow/mxnet/jax model class.
- Extended models: These consist of a core model and an encoder.
"""

__all__ = [
    # Sub-Packages
    "activations",
    "generic",
    "pretrained",
    # Constants
    "MODELS",
    # ABCs & Protocols
    "ForecastingModel",
    "StateSpaceForecastingModel",
    "BaseModel",
    # Classes
    "ODE_RNN",
    "SetFuncTS",
    "GroupedSetFuncTS",
]

from . import activations, generic, pretrained
from .base import BaseModel, ForecastingModel, StateSpaceForecastingModel
from .ode_rnn import ODE_RNN
from .set_function_for_timeseries import GroupedSetFuncTS, SetFuncTS

MODELS: dict[str, type[BaseModel]] = {
    "ODE_RNN": ODE_RNN,
}
r"""Dictionary of all available models."""
