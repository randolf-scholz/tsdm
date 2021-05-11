from collections import Mapping

import numpy as np
import torch
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike
from scipy.stats import mode
from torch import nn, Tensor
from matplotlib.axes import Axes

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
    r"""
    Scaled $\ell^p$-norm: :math:`\|x\|_p = \big(\tfrac{1}{n}\sum_{i=1}^n x_i^p \big)^{1/p}`

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
    kwargs = {axis_name : axis, keep_name: keepdims}

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


def visualize_distribution(x: ArrayLike, ax: Axes, bins=50, log=True, loc="upper right", print_stats=True) -> None:
    r"""
    Plots the distribution of x in the given axis.

    Parameters
    ----------
    x: ArrayLike
    ax: Axes
    bins: int
    log: bool
    loc: string
    print_stats: bool
    """
    if type(x) == Tensor:
        x = x.detach().cpu().numpy()
    else:
        x = np.array(x)

    x = x.flatten().astype(float)
    nans = np.isnan(x)
    x = x[~nans]

    ax.grid(axis='x')
    ax.set_axisbelow(True)

    if log:
        tol = 2**-24 if x.dtype == np.float32 else 2**-53
        z = np.log10(np.maximum(x, tol))
        ax.set_xscale('log')
        ax.set_yscale('log')
        low = np.floor(np.quantile(z, 0.01))
        high = np.ceil(np.quantile(z, 1 - 0.01))
        x = x[(z >= low) & (z <= high)]
        bins = np.logspace(low, high, num=bins, base=10)

    ax.hist(x, bins=bins, density=True)

    if print_stats:
        text = r"\begin{tabular}{ll}"\
               + F"NaNs   & {100 * np.mean(nans):.2f}"+r"\%" + r" \\ "\
               + F"Mean   & {np.mean(x):.2e}" + r" \\ "\
               + F"Median & {np.median(x):.2e}" + r" \\ "\
               + F"Mode   & {mode(x)[0][0]:.2e}" + r" \\ "\
               + F"stdev  & {np.std(x):.2e}" + r" \\ "\
               + r"\end{tabular}"
        textbox = AnchoredText(text, loc=loc)
        ax.add_artist(textbox)
