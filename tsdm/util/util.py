import gc
import logging
from collections import Mapping
from functools import wraps
from time import perf_counter_ns
from typing import Callable

import numpy as np
import torch

from numpy.typing import ArrayLike
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


def scaled_norm(x, p=2, axis=None, keepdims=False) -> ArrayLike:
    r"""Scaled $\ell^p$-norm, works with both :class:`torch.Tensor` and :class:`numpy.ndarray`

    .. math::
        \|x\|_p = \big(\tfrac{1}{n}\sum_{i=1}^n x_i^p \big)^{1/p}

    Parameters
    ----------
    x: ArrayLike
    p: int=2
    axis: tuple[int]=None
    keepdims: bool=False

    Returns
    -------
    ArrayLike
    """
    fwork = torch if type(x) == Tensor else np
    x = fwork.abs(x)

    axis_name = {torch: 'dim', np: 'axis'}[fwork]
    keep_name = {torch: 'keepdim', np: 'keepdims'}[fwork]
    # only pass arguments if axis is specified
    kwargs = {} if axis is None else {axis_name: axis, keep_name: keepdims}

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return fwork.exp(fwork.mean(fwork.log(x), **kwargs))
    elif p == 1:
        return fwork.mean(x, **kwargs)
    elif p == 2:
        return fwork.sqrt(fwork.mean(x**2, **kwargs))
    elif p == float('inf'):
        return fwork.max(x, **kwargs)
    else:
        return fwork.mean(x**p, **kwargs)**(1/p)


def relative_error(xhat: ArrayLike, x_true: ArrayLike, eps=None) -> ArrayLike:
    R"""Relative error, works with both :class:`torch.Tensor` and :class:`numpy.ndarray`

    .. math::
        \operatorname{r}(\hat x, x) = \tfrac{|\hat x - x|}{|x|+\epsilon}

    Parameters
    ----------
    xhat: ArrayLike
        The estimation
    x_true:  ArrayLike
        The true value
    eps: float, default=None
        Tolerance parameter. By default, $2^{-24}$ for single and $2^{-53}$ for double precision

    Returns
    -------
    ArrayLike
    """
    fwork = torch if type(xhat) == Tensor else np

    if xhat.dtype in (fwork.float32, fwork.int32):
        _eps = 2**-24
    elif xhat.dtype in (fwork.float64, fwork.int64):
        _eps = 2**-53
    else:
        _eps = 2**-24

    eps = eps or _eps
    r = fwork.abs(xhat-x_true)/(fwork.abs(x_true) + eps)
    return r


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
            logger.log(loglevel, F"{fun.__name__} executed in {elapsed:.4f}s")
        except Exception:
            result = None
            elapsed = float('nan')
            logger.log(loglevel, F"{fun.__name__} failed with {Exception=}")
        gc.enable()

        if append:
            return result, elapsed
        else:
            return result

    return timed_fun
