r"""Implementation / loading mechanism for models.

There are two types of models:

- Core models: These consist of only a pytorch/tensorflow/mxnet/jax model class.
- Extended models: These consist of a core model and an encoder.
"""

__all__ = [
    # Sub-Packages
    "pretrained",
    # ABCs & Protocols
    "ForecastingModel",
    "StateSpaceForecastingModel",
    "BaseModel",
]

from . import pretrained
from .base import BaseModel, ForecastingModel, StateSpaceForecastingModel
