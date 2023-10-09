"""Tensor functions."""

__all__ = [
    # Functions
    "geometric_mean",
    "grad_norm",
    "multi_norm",
    "norm",
    "relative_error",
    "scaled_norm",
    "tensor_norm",
]


from typing import Union

import torch
from torch import Tensor, jit


@jit.script
def relative_error(xhat: Tensor, x_true: Tensor) -> Tensor:
    r"""Element-wise relative error of `xhat` w.r.t. `x_true`.

    .. Signature:: ``[(..., n), (..., n)] -> (..., n)``

    Note:
        Automatically adds a small epsilon to the denominator to avoid division by zero.
    """
    EPS: dict[torch.dtype, float] = {
        torch.float16: 1e-3,
        torch.float32: 1e-6,
        torch.float64: 1e-15,
        # complex floats
        torch.complex32: 1e-3,
        torch.complex64: 1e-6,
        torch.complex128: 1e-15,
        # other floats
        torch.bfloat16: 1e-2,
    }
    eps = EPS[xhat.dtype]
    return torch.abs(xhat - x_true) / (torch.abs(x_true) + eps)


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
def scaled_norm(
    x: Tensor,
    p: float = 2.0,
    axis: Union[None, int, list[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Shortcut for scaled norm.

    .. Signature:: ``(..., n) -> ...``
    """
    # TODO: deal with nan values
    x = x.abs()

    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    if p == float("inf"):
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -float("inf"):
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return geometric_mean(x, axis=dim, keepdim=keepdim)

    # NOTE: preconditioning with x_max is not necessary, but it helps with numerical stability and prevents overflow
    x_max = x.abs().amax(dim=dim, keepdim=True)
    result = x_max * (x / x_max).pow(p).mean(dim=dim, keepdim=True).pow(1 / p)
    return result.squeeze(dim=dim * (1 - int(keepdim)))  # branchless
    # return x.pow(p).mean(dim=dim, keepdim=keepdim).pow(1 / p)


@jit.script
def norm(
    x: Tensor,
    p: float = 2.0,
    axis: Union[None, int, list[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Shortcut for non-scaled norm.

    .. Signature:: ``(..., n) -> ...``

    Only present here for testing purposes.
    """
    # TODO: deal with nan values
    x = x.abs()

    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    # non-scaled
    if p == float("inf"):
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -float("inf"):
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return (x != 0).sum(dim=dim, keepdim=keepdim)

    # NOTE: preconditioning improves numerical stability
    x_max = x.amax(dim=dim, keepdim=True)
    result = x_max * (x / x_max).pow(p).sum(dim=dim, keepdim=True).pow(1 / p)
    return result.squeeze(dim=dim * (1 - int(keepdim)))  # branchless
    # return x.pow(p).sum(dim=dim, keepdim=keepdim).pow(1 / p)


@jit.script
def tensor_norm(
    x: Tensor,
    p: float = 2.0,
    axis: Union[None, int, list[int]] = None,
    keepdim: bool = False,
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
    return (
        scaled_norm(x, p=p, axis=axis, keepdim=keepdim)
        if scaled
        else norm(x, p=p, axis=axis, keepdim=keepdim)
    )


@jit.script
def multi_norm(
    tensors: list[Tensor],
    p: float = 2,
    q: float = 2,
    scaled: bool = True,
) -> Tensor:
    r"""Return the (scaled) p-q norm of the gradients.

    .. Signature:: ``(...) -> ()``


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

    x = tensors[0]
    s = tensor_norm(x, p=p, scaled=scaled) ** q
    for x in tensors[1:]:
        s += tensor_norm(x, p=p, scaled=scaled) ** q
    return (s / (1 + int(scaled) * len(tensors))) ** (1 / q)


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

    # Initializing s this way automatically gets the dtype and device correct
    x = tensors[0]
    assert x.grad is not None
    s = tensor_norm(x.grad, p=p, scaled=scaled) ** q
    for x in tensors[1:]:
        assert x.grad is not None
        s += tensor_norm(x.grad, p=p, scaled=scaled) ** q
    return (s / (1 + int(scaled) * len(tensors))) ** (1 / q)
