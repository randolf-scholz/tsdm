r"""Implementations of activation functions.

Notes
-----
Contains activations in modular form.
  - See `tsdm.util.activations.functional` for functional implementations.
"""

__all__ = [
    # Constants
    "ModularActivation",
    "ModularActivations",
]

import logging
from typing import Final

from torch import nn

__logger__ = logging.getLogger(__name__)

ModularActivation = nn.Module
r"""Type hint for modular activations."""

ModularActivations: Final[dict[str, type[nn.Module]]] = {
    "AdaptiveLogSoftmaxWithLoss": nn.AdaptiveLogSoftmaxWithLoss,
    "ELU": nn.ELU,
    "Hardshrink": nn.Hardshrink,
    "Hardsigmoid": nn.Hardsigmoid,
    "Hardtanh": nn.Hardtanh,
    "Hardswish": nn.Hardswish,
    "Identity": nn.Identity,
    "LeakyReLU": nn.LeakyReLU,
    "LogSigmoid": nn.LogSigmoid,
    "LogSoftmax": nn.LogSoftmax,
    "MultiheadAttention": nn.MultiheadAttention,
    "PReLU": nn.PReLU,
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "SiLU": nn.SiLU,
    "Softmax": nn.Softmax,
    "Softmax2d": nn.Softmax2d,
    "Softplus": nn.Softplus,
    "Softshrink": nn.Softshrink,
    "Softsign": nn.Softsign,
    "Tanh": nn.Tanh,
    "Tanhshrink": nn.Tanhshrink,
    "Threshold": nn.Threshold,
}
r"""Dictionary of all available modular activations."""