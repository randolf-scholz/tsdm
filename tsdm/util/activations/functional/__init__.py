r"""Implementations of activation functions.

Notes
-----
Contains activations in functional form.
  - See :mod:`tsdm.util.activations.modular` for modular implementations.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "FunctionalActivation",
    "FunctionalActivations",
]

import logging
from typing import Callable, Final

from torch import Tensor
from torch.nn import functional as F

from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

FunctionalActivation = Callable[..., Tensor]
r"""Type hint for functional activations."""

FunctionalActivations: Final[LookupTable[FunctionalActivation]] = {
    "threshold": F.threshold,
    # Thresholds each element of the input Tensor.
    "threshold_": F.threshold_,  # type: ignore[attr-defined]
    # type: ignore  # In-place version of threshold().
    "relu": F.relu,
    # Applies the rectified linear unit function element-wise.
    "relu_": F.relu_,
    # In-place version of relu().
    "hardtanh": F.hardtanh,
    # Applies the HardTanh function element-wise.
    "hardtanh_": F.hardtanh_,
    # In-place version of hardtanh().
    "hardswish": F.hardswish,
    # Applies the hardswish function, element-wise, as described in the paper:
    "relu6": F.relu6,
    # Applies the element-wise function `ReLU6(x)=\min(\max(0,x),6)`.
    "elu": F.elu,
    # Applies element-wise, `ELU(x)=\max(0,x)+\min(0,α⋅(\exp(x)−1))`.
    "elu_": F.elu_,
    # In-place version of elu().
    "selu": F.selu,
    # Applies element-wise, `SELU(x)=β⋅(\max(0,x)+\min(0,α⋅(e^x−1)))` with `α≈1.677` and `β≈1.05`.
    "celu": F.celu,
    # Applies element-wise, `CELU(x)= \max(0,x)+\min(0,α⋅(\exp(x/α)−1)`.
    "leaky_relu": F.leaky_relu,
    # Applies element-wise, `LeakyReLU(x)=\max(0,x)+negative_slope⋅\min(0,x)`.
    "leaky_relu_": F.leaky_relu_,
    # In-place version of leaky_relu().
    "prelu": F.prelu,
    # `PReLU(x)=\max(0,x)+ω⋅\min(0,x)` where ω is a learnable parameter.
    "rrelu": F.rrelu,
    # Randomized leaky ReLU.
    "rrelu_": F.rrelu_,
    # In-place version of rrelu().
    "glu": F.glu,
    # The gated linear unit.
    "gelu": F.gelu,
    # Applies element-wise the function `GELU(x)=x⋅Φ(x)`.
    "logsigmoid": F.logsigmoid,
    # Applies element-wise `LogSigmoid(x_i)=\log(1/(1+\exp(−x_i)))`.
    "hardshrink": F.hardshrink,
    # Applies the hard shrinkage function element-wise.
    "tanhshrink": F.tanhshrink,
    # Applies element-wise, `Tanhshrink(x)=x−\tanh(x)`.
    "softsign": F.softsign,
    # Applies element-wise, the function `SoftSign(x)=x/(1+∣x∣)`.
    "softplus": F.softplus,
    # Applies element-wise, the function `Softplus(x)=1/β⋅\log(1+\exp(β⋅x))`.
    "softmin": F.softmin,
    # Applies a softmin function.
    "softmax": F.softmax,
    # Applies a softmax function.
    "softshrink": F.softshrink,
    # Applies the soft shrinkage function elementwise
    "gumbel_softmax": F.gumbel_softmax,
    # Samples from the Gumbel-Softmax distribution and optionally discretizes.
    "log_softmax": F.log_softmax,
    # Applies a softmax followed by a logarithm.
    "tanh": F.tanh,
    # Applies element-wise, `\tanh(x)=(\exp(x)−\exp(−x))/(\exp(x)+\exp(−x))`.
    "sigmoid": F.sigmoid,
    # Applies the element-wise function `Sigmoid(x)=1/(1+\exp(−x))`.
    "hardsigmoid": F.hardsigmoid,
    # Applies the element-wise function.
    "silu": F.silu,
    # Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    "mish": F.mish,
    # Applies the Mish function, element-wise.
    "batch_norm": F.batch_norm,
    # Applies Batch Normalization for each channel across a batch of data.
    "group_norm": F.group_norm,
    # Applies Group Normalization for last certain number of dimensions.
    "instance_norm": F.instance_norm,
    # Applies Instance Normalization for each channel in each data sample in a batch.  # noqa
    "layer_norm": F.layer_norm,
    # Applies Layer Normalization for last certain number of dimensions.
    "local_response_norm": F.local_response_norm,
    # Applies local response normalization over an input signal composed of several input planes.
    "normalize": F.normalize,
    # Performs Lp normalization of inputs over specified dimension.
}
r"""Dictionary of all available functional activations."""
