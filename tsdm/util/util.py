r"""
Utility functions
"""


import gc
import logging
from collections.abc import Mapping
from functools import wraps
from time import perf_counter_ns
from typing import Callable, Union

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
import torch
from torch import nn, Tensor


ACTIVATIONS = {
    'AdaptiveLogSoftmaxWithLoss' : nn.AdaptiveLogSoftmaxWithLoss,
    'ELU'         : nn.ELU,
    'Hardshrink'  : nn.Hardshrink,
    'Hardsigmoid' : nn.Hardsigmoid,
    'Hardtanh'    : nn.Hardtanh,
    'Hardswish'   : nn.Hardswish,
    'LeakyReLU'   : nn.LeakyReLU,
    'LogSigmoid'  : nn.LogSigmoid,
    'LogSoftmax'  : nn.LogSoftmax,
    'MultiheadAttention' : nn.MultiheadAttention,
    'PReLU'       : nn.PReLU,
    'ReLU'        : nn.ReLU,
    'ReLU6'       : nn.ReLU6,
    'RReLU'       : nn.RReLU,
    'SELU'        : nn.SELU,
    'CELU'        : nn.CELU,
    'GELU'        : nn.GELU,
    'Sigmoid'     : nn.Sigmoid,
    'SiLU'        : nn.SiLU,
    'Softmax'     : nn.Softmax,
    'Softmax2d'   : nn.Softmax2d,
    'Softplus'    : nn.Softplus,
    'Softshrink'  : nn.Softshrink,
    'Softsign'    : nn.Softsign,
    'Tanh'        : nn.Tanh,
    'Tanhshrink'  : nn.Tanhshrink,
    'Threshold'   : nn.Threshold,
}


def _torch_is_float_dtype(x: Tensor) -> bool:
    return x.dtype in (torch.half, torch.float, torch.double, torch.bfloat16,
                       torch.complex32, torch.complex64, torch.complex128)


def deep_dict_update(d: dict, new: Mapping) -> dict:
    r"""
    Updates nested dictionary recursively in-place with new dictionary

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new: Mapping
    """

    for key, value in new.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            d[key] = new[key]
    return d


def deep_kval_update(d: dict, **new_kv) -> dict:
    r"""
    Updates nested dictionary recursively in-place with key-value pairs

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kv: Mapping
    """

    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_kval_update(d.get(key, {}), **new_kv)
        elif key in new_kv:
            d[key] = new_kv[key]
    return d


def relative_error(xhat: Union[ArrayLike, Tensor], x_true: Union[ArrayLike, Tensor]
                   ) -> Union[ArrayLike, Tensor]:
    R"""Relative error, works with both :class:`~torch.Tensor` and :class:`~numpy.ndarray`

    .. math::
        \operatorname{r}(\hat x, x) = \tfrac{|\hat x - x|}{|x|+\epsilon}

    The tolerance parameter $\epsilon$ is determined automatically. By default,
    $\epsilon=2^{-24}$ for single and $\epsilon=2^{-53}$ for double precision.

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
    if isinstance(xhat, Tensor) and isinstance(x_true, Tensor):
        return _torch_relative_error(xhat, x_true)

    xhat, x_true = np.asanyarray(xhat), np.asanyarray(x_true)
    return _numpy_relative_error(xhat, x_true)


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


def _torch_relative_error(xhat: Tensor, x_true: Tensor) -> Tensor:
    if xhat.dtype == torch.bfloat16:
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


def _torch_scaled_norm(x: Tensor,  p: float = 2, dim: Union[int, tuple[int, ...]] = None,
                       keepdim: bool = False) -> Tensor:
    dim = () if dim is None else dim

    if not _torch_is_float_dtype(x):
        x = x.to(dtype=torch.float)
    x = torch.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return torch.exp(torch.mean(torch.log(x), dim=dim, keepdim=keepdim))
    if p == 1:
        return torch.mean(x, dim=dim, keepdim=keepdim)
    if p == 2:
        return torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=keepdim))
    if p == float('inf'):
        return torch.amax(x, dim=dim, keepdim=keepdim)
    # other p
    return torch.mean(x**p, dim=dim, keepdim=keepdim) ** (1 / p)


def _numpy_scaled_norm(x: ndarray, p: float = 2, axis: Union[int, tuple[int, ...]] = None,
                       keepdims: bool = False) -> ndarray:
    x = np.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return np.exp(np.mean(np.log(x), axis=axis, keepdims=keepdims))
    if p == 1:
        return np.mean(x, axis=axis, keepdims=keepdims)
    if p == 2:
        return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=keepdims))
    if p == float('inf'):
        return np.max(x, axis=axis, keepdims=keepdims)
    # other p
    return np.mean(x**p, axis=axis, keepdims=keepdims) ** (1 / p)


def scaled_norm(x: Union[ArrayLike, Tensor], p: float = 2,
                axis: Union[None, int, tuple[int, ...]] = None,
                keepdims: bool = False) -> Union[ArrayLike, Tensor]:
    r"""Scaled $\ell^p$-norm, works with both :class:`torch.Tensor` and :class:`numpy.ndarray`

    .. math::
        \|x\|_p = \big(\tfrac{1}{n}\sum_{i=1}^n x_i^p \big)^{1/p}

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
    if isinstance(x, Tensor):
        return _torch_scaled_norm(x, p=p, dim=axis, keepdim=keepdims)

    x = np.asanyarray(x)
    return _numpy_scaled_norm(x, p=p, axis=axis, keepdims=keepdims)


def timefun(fun: Callable, append=True, loglevel: int = logging.WARNING) -> Callable:
    r"""Logs the execution time of the function. Use as decorator.

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

    logger = logging.getLogger('timefun')

    @wraps(fun)
    def timed_fun(*args, **kwargs):
        gc.collect()
        gc.disable()
        try:
            start_time = perf_counter_ns()
            result = fun(*args, **kwargs)
            end_time = perf_counter_ns()
            elapsed = (end_time - start_time) / 10 ** 9
            logger.log(loglevel, "%s executed in %.4f s" , fun.__name__, elapsed)
        except (KeyboardInterrupt, SystemExit) as E:
            raise E
        except Exception as E:  # pylint: disable=W0703
            result = None
            elapsed = float('nan')
            RuntimeWarning(F"Function execution failed with Exception {E}")
            logger.log(loglevel, "%s failed with Exception %s" , fun.__name__, E)
        gc.enable()

        if append:
            return result, elapsed
        # else
        return result

    return timed_fun
