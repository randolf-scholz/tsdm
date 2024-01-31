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
    "Model",
    "BaseModel",
    # Classes
    "ODE_RNN",
    "SetFuncTS",
    "GroupedSetFuncTS",
]
from tsdm.models import activations, generic, pretrained
from tsdm.models._models import (
    BaseModel,
    ForecastingModel,
    Model,
    StateSpaceForecastingModel,
)
from tsdm.models.ode_rnn import ODE_RNN
from tsdm.models.set_function_for_timeseries import GroupedSetFuncTS, SetFuncTS

MODELS: dict[str, type[Model]] = {
    "ODE_RNN": ODE_RNN,
    "SetFuncTS": SetFuncTS,
}
r"""Dictionary of all available models."""
