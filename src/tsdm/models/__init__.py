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
    # Type Hints
    "Model",
    "ModelType",
    # Constants
    "MODELS",
    # Classes
    "BaseModel",
    "ODE_RNN",
    "SetFuncTS",
    "GroupedSetFuncTS",
]

from torch import nn
from typing_extensions import Final, TypeAlias

from tsdm.models import activations, generic, pretrained
from tsdm.models._models import BaseModel
from tsdm.models.ode_rnn import ODE_RNN
from tsdm.models.set_function_for_timeseries import GroupedSetFuncTS, SetFuncTS

Model: TypeAlias = nn.Module
r"""Type hint for models."""

ModelType = type[nn.Module]
r"""Type hint for models."""

MODELS: Final[dict[str, ModelType]] = {
    "ODE_RNN": ODE_RNN,
    "SetFuncTS": SetFuncTS,
}
r"""Dictionary of all available models."""

del Final, TypeAlias, nn
