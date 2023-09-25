"""Tensor functions."""

__all__ = [
    "geometric_mean",
    "grad_norm",
    "multi_norm",
    "tensor_norm",
    "scaled_norm",
]


from typing import Union

import torch
from torch import Tensor, jit


@jit.script
def geometric_mean(
    x: Tensor,
    axis: Union[None, int, list[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Geometric mean of a tensor.

    .. Signature:: ``(..., n) -> (...)``
    """
    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    return x.log().nanmean(dim=dim, keepdim=keepdim).exp()


@jit.script
def multi_norm(
    tensors: list[Tensor],
    p: float = 2,
    q: float = 2,
    scaled: bool = True,
) -> Tensor:
    r"""Return the (scaled) p-q norm of the gradients.

    .. math:: ‖A‖_{p,q} ≔ \Bigl|∑_{j=1}^n \Big(∑_{i=1}^m |A_{ij}|^p\Big)^{q/p}\Bigr|^{1/q}

    If `normalize=True`, the sums are replaced with averages.
    """
    _tensors: list[Tensor] = []
    for tensor in tensors:
        if tensor.numel() > 0:
            _tensors.append(tensor)
    tensors = _tensors

    if len(tensors) == 0:
        return torch.tensor(0.0)

    if scaled:
        # Initializing s this way automatically gets the dtype and device correct
        s = torch.mean(tensors.pop() ** p) ** (q / p)
        for x in tensors:
            s += torch.mean(x**p) ** (q / p)
        return (s / (1 + len(tensors))) ** (1 / q)

    # else
    s = torch.sum(tensors.pop() ** p) ** (q / p)
    for x in tensors:
        s += torch.sum(x**p) ** (q / p)
    return s ** (1 / q)


@jit.script
def grad_norm(
    tensors: list[Tensor],
    p: float = 2,
    q: float = 2,
    scaled: bool = True,
) -> Tensor:
    r"""Return the (scaled) p-q norm of the gradients.

    .. math:: ‖A‖_{p,q} ≔ \Bigl|∑_{j=1}^n \Big(∑_{i=1}^m |A_{ij}|^p\Big)^{q/p}\Bigr|^{1/q}

    If `normalize=True`, the sums are replaced with averages.
    """
    if len(tensors) == 0:
        return torch.tensor(0.0)

    if scaled:
        # Initializing s this way automatically gets the dtype and device correct
        x = tensors.pop()
        assert x.grad is not None
        s = torch.mean(x.grad**p) ** (q / p)
        for x in tensors:
            assert x.grad is not None
            s += torch.mean(x.grad**p) ** (q / p)
        return (s / (1 + len(tensors))) ** (1 / q)
    # else
    x = tensors.pop()
    assert x.grad is not None
    s = torch.sum(x.grad**p) ** (q / p)
    for x in tensors:
        assert x.grad is not None
        s += torch.sum(x.grad**p) ** (q / p)
    return s ** (1 / q)


@jit.script
def tensor_norm(
    x: Tensor,
    p: float = 2.0,
    axis: Union[None, int, list[int]] = None,
    keepdim: bool = True,
    scaled: bool = False,
) -> Tensor:
    r"""Entry-wise norm of $p$-th order.

    .. Signature:: ``(..., n) -> ...``

    +--------+-----------------------------------+------------------------------------+
    |        | standard                          | size normalized                    |
    +========+===================================+====================================+
    | $p=+∞$ | maximum value                     | maximum value                      |
    +--------+-----------------------------------+------------------------------------+
    | $p=+2$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=+1$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=±0$ | ∞ or sum of non-zero values       | geometric mean of values           |
    +--------+-----------------------------------+------------------------------------+
    | $p=-1$ | reciprocal sum of absolute values | reciprocal mean of absolute values |
    +--------+-----------------------------------+------------------------------------+
    | $p=-2$ | reciprocal sum of squared values  | reciprocal mean of squared values  |
    +--------+-----------------------------------+------------------------------------+
    | $p=-∞$ | minimum value                     | minimum value                      |
    +--------+-----------------------------------+------------------------------------+
    """
    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)
    x = x.abs()

    # TODO: deal with nan values

    if scaled:
        if p == float("inf"):
            return x.amax(dim=dim, keepdim=keepdim)
        if p == -float("inf"):
            return x.amin(dim=dim, keepdim=keepdim)
        if p == 0:
            return geometric_mean(x, axis=dim, keepdim=keepdim)

        # NOTE: preconditioning improves numerical stability
        x_max = x.amax(dim=dim, keepdim=True)
        return x_max * (x / x_max).pow(p).mean(dim=dim, keepdim=keepdim).pow(1 / p)

    # non-scaled
    if p == float("inf"):
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -float("inf"):
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return (x != 0).sum(dim=dim, keepdim=keepdim)

    # NOTE: preconditioning improves numerical stability
    x_max = x.amax(dim=dim, keepdim=True)
    return x_max * (x / x_max).pow(p).sum(dim=dim, keepdim=keepdim).pow(1 / p)


@jit.script
def scaled_norm(
    x: Tensor,
    p: float = 2.0,
    axis: Union[None, int, list[int]] = None,
    keepdim: bool = True,
) -> Tensor:
    r"""Shortcut for `tensor_norm(x, p, axes, keepdim, scaled=True)`.

    .. Signature:: ``(..., n) -> ...``
    """
    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)
    x = x.abs()

    # TODO: deal with nan values

    if p == float("inf"):
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -float("inf"):
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return geometric_mean(x, axis=dim, keepdim=keepdim)

    # NOTE: preconditioning with x_max is not necessary, but it helps with numerical stability
    x_max = x.amax(dim=dim, keepdim=True)
    return x_max * (x / x_max).pow(p).mean(dim=dim, keepdim=keepdim).pow(1 / p)
