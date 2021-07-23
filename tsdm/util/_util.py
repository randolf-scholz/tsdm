r"""Utility functions."""
import gc
import logging
from collections.abc import Mapping
from functools import singledispatch, wraps
from time import perf_counter_ns
from typing import Callable, Union, Final

# from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from numpy import ndarray
from numpy.typing import ArrayLike
from torch import Tensor, nn

logger = logging.getLogger(__name__)
__all__ = [
    "ACTIVATIONS",
    "OPTIMIZERS",
    "deep_dict_update",
    "deep_kval_update",
    "relative_error",
    "scaled_norm",
    "timefun",
]

ACTIVATIONS: Final[dict[str, nn.Module]] = {
    # Utility dictionary, for use in model creation from Hyperparameter dicts
    "AdaptiveLogSoftmaxWithLoss": nn.AdaptiveLogSoftmaxWithLoss,
    "ELU": nn.ELU,
    "Hardshrink": nn.Hardshrink,
    "Hardsigmoid": nn.Hardsigmoid,
    "Hardtanh": nn.Hardtanh,
    "Hardswish": nn.Hardswish,
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

OPTIMIZERS: Final[dict[str, Optimizer]] = {
    # Utility dictionary, for use in model creation from Hyperparameter dicts
    "Adadelta": torch.optim.Adadelta,
    # Implements Adadelta algorithm.
    "Adagrad": torch.optim.Adagrad,
    # Implements Adagrad algorithm.
    "Adam": torch.optim.Adam,
    # Implements Adam algorithm.
    "AdamW": torch.optim.AdamW,
    # Implements AdamW algorithm.
    "SparseAdam": torch.optim.SparseAdam,
    # Implements lazy version of Adam algorithm suitable for sparse tensors.
    "Adamax": torch.optim.Adamax,
    # Implements Adamax algorithm (a variant of Adam based on infinity norm).
    "ASGD": torch.optim.ASGD,
    # Implements Averaged Stochastic Gradient Descent.
    "LBFGS": torch.optim.LBFGS,
    # Implements L-BFGS algorithm, heavily inspired by minFunc.
    "RMSprop": torch.optim.RMSprop,
    # Implements RMSprop algorithm.
    "Rprop": torch.optim.Rprop,
    # Implements the resilient backpropagation algorithm.
    "SGD": torch.optim.SGD,
    # Implements stochastic gradient descent (optionally with momentum).
}


def _torch_is_float_dtype(x: Tensor) -> bool:
    return x.dtype in (
        torch.half,
        torch.float,
        torch.double,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
    )


def deep_dict_update(d: dict, new_kvals: Mapping) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: Mapping
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in new_kvals.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


def deep_kval_update(d: dict, **new_kvals: dict) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: dict
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_kval_update(d.get(key, {}), **new_kvals)
        elif key in new_kvals:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


@singledispatch
def relative_error(
    xhat: Union[ArrayLike, Tensor], x_true: Union[ArrayLike, Tensor]
) -> Union[ArrayLike, Tensor]:
    r"""Relative error, works with both :class:`~torch.Tensor` and :class:`~numpy.ndarray`.

    .. math::
        r(x̂, x) = \tfrac{|x̂ - x|}{|x|+ϵ}

    The tolerance parameter $ϵ$ is determined automatically. By default,
    $ϵ=2^{-24}$ for single and $ϵ=2^{-53}$ for double precision.

    Parameters
    ----------
    xhat: ArrayLike
        The estimation
    x_true:  ArrayLike
        The true value

    Returns
    -------
    ArrayLike
    """
    xhat, x_true = np.asanyarray(xhat), np.asanyarray(x_true)
    return _numpy_relative_error(xhat, x_true)


@relative_error.register
def _numpy_relative_error(xhat: ndarray, x_true: ndarray) -> ndarray:
    if xhat.dtype in (np.float16, np.int16):
        eps = 2 ** -11
    elif xhat.dtype in (np.float32, np.int32):
        eps = 2 ** -24
    elif xhat.dtype in (np.float64, np.int64):
        eps = 2 ** -53
    else:
        raise NotImplementedError

    return np.abs(xhat - x_true) / (np.abs(x_true) + eps)


@relative_error.register
def _torch_relative_error(xhat: Tensor, x_true: Tensor) -> Tensor:
    if xhat.dtype in (torch.bfloat16,):
        eps = 2 ** -8
    elif xhat.dtype in (torch.float16, torch.int16):
        eps = 2 ** -11
    elif xhat.dtype in (torch.float32, torch.int32):
        eps = 2 ** -24
    elif xhat.dtype in (torch.float64, torch.int64):
        eps = 2 ** -53
    else:
        raise NotImplementedError

    # eps = eps or _eps
    return torch.abs(xhat - x_true) / (torch.abs(x_true) + eps)


@singledispatch
def scaled_norm(
    x: Union[ArrayLike, Tensor],
    p: float = 2,
    axis: Union[None, int, tuple[int, ...]] = None,
    keepdims: bool = False,
) -> Union[ArrayLike, Tensor]:
    r"""Scaled $ℓ^p$-norm, works with both :class:`torch.Tensor` and :class:`numpy.ndarray`.

    .. math::
        ‖x‖_p = (⅟ₙ\sum_{i=1}^n x_i^p)^{1/p}

    Parameters
    ----------
    x: ArrayLike
    p: int, default=2
    axis: tuple[int], default=None
    keepdims: bool, default=False

    Returns
    -------
    ArrayLike
    """
    x = np.asanyarray(x)
    return scaled_norm(x, p=p, axis=axis, keepdims=keepdims)


@scaled_norm.register
def _torch_scaled_norm(
    x: Tensor,
    p: float = 2,
    axis: Union[int, tuple[int, ...]] = None,
    keepdims: bool = False,
) -> Tensor:
    axis = () if axis is None else axis

    if not _torch_is_float_dtype(x):
        x = x.to(dtype=torch.float)
    x = torch.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return torch.exp(torch.mean(torch.log(x), dim=axis, keepdim=keepdims))
    if p == 1:
        return torch.mean(x, dim=axis, keepdim=keepdims)
    if p == 2:
        return torch.sqrt(torch.mean(x ** 2, dim=axis, keepdim=keepdims))
    if p == float("inf"):
        return torch.amax(x, dim=axis, keepdim=keepdims)
    # other p
    return torch.mean(x ** p, dim=axis, keepdim=keepdims) ** (1 / p)


@scaled_norm.register
def _numpy_scaled_norm(
    x: ndarray,
    p: float = 2,
    axis: Union[int, tuple[int, ...]] = None,
    keepdims: bool = False,
) -> ndarray:
    x = np.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return np.exp(np.mean(np.log(x), axis=axis, keepdims=keepdims))
    if p == 1:
        return np.mean(x, axis=axis, keepdims=keepdims)
    if p == 2:
        return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=keepdims))
    if p == float("inf"):
        return np.max(x, axis=axis, keepdims=keepdims)
    # other p
    return np.mean(x ** p, axis=axis, keepdims=keepdims) ** (1 / p)


def timefun(
    fun: Callable, append: bool = True, loglevel: int = logging.WARNING
) -> Callable:
    r"""Log the execution time of the function. Use as decorator.

    By default appends the execution time (in seconds) to the function call.

    ``outputs, time_elapse = timefun(f, append=True)(inputs)``

    If the function call failed, ``outputs=None`` and ``time_elapsed=float('nan')`` are returned.

    Parameters
    ----------
    fun: Callable
    append: bool, default=True
        Whether to append the time result to the function call
    loglevel: int, default=logging.Warning (20)
    """
    timefunlogger = logging.getLogger("timefun")

    @wraps(fun)
    def timed_fun(*args, **kwargs):
        gc.collect()
        gc.disable()
        try:
            start_time = perf_counter_ns()
            result = fun(*args, **kwargs)
            end_time = perf_counter_ns()
            elapsed = (end_time - start_time) / 10 ** 9
            timefunlogger.log(loglevel, "%s executed in %.4f s", fun.__name__, elapsed)
        except (KeyboardInterrupt, SystemExit) as E:
            raise E
        except Exception as E:  # pylint: disable=W0703
            result = None
            elapsed = float("nan")
            RuntimeWarning(f"Function execution failed with Exception {E}")
            timefunlogger.log(loglevel, "%s failed with Exception %s", fun.__name__, E)
        gc.enable()

        if append:
            return result, elapsed
        # else
        return result

    return timed_fun
