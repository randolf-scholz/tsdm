r"""Tensor functions."""

__all__ = [
    # Functions
    "geometric_mean",
    "grad_norm",
    "multi_norm",
    "norm",
    "scaled_norm",
    "tensor_norm",
]


import torch
from torch import Tensor


@torch.compile(fullgraph=True)
def geometric_mean(
    x: Tensor,
    /,
    *,
    axis: None | int | list[int] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Geometric mean of a tensor.

    .. signature:: ``(..., n) -> (...)``
    """
    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    return x.log().nanmean(dim=dim, keepdim=keepdim).exp()


@torch.compile(fullgraph=True)
def scaled_norm(
    x: Tensor,
    /,
    *,
    p: float = 2.0,
    axis: None | int | list[int] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Shortcut for scaled norm.

    .. signature:: ``(..., n) -> ...``
    """
    # TODO: deal with nan values
    x = x.abs()

    if axis is None:
        dim = list(range(x.ndim))
    elif isinstance(axis, int):
        dim = [axis]
    else:
        dim = axis

    if p == torch.inf:
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -torch.inf:
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return geometric_mean(x, axis=dim, keepdim=keepdim)

    # NOTE: preconditioning with x_max is not necessary, but it helps with numerical stability and prevents overflow
    x_max = x.abs().amax(dim=dim, keepdim=True)
    result = x_max * (x / x_max).pow(p).mean(dim=dim, keepdim=True).pow(1 / p)
    return result.squeeze(dim=dim * (1 - int(keepdim)))  # branchless


@torch.compile(fullgraph=True)
def norm(
    x: Tensor,
    /,
    *,
    p: float = 2.0,
    axis: None | int | list[int] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Shortcut for non-scaled norm.

    .. signature:: ``(..., n) -> ...``

    Only present here for testing purposes.
    """
    # TODO: deal with nan values
    x = x.abs()

    dim: list[int] = (
        list(range(x.ndim))
        if axis is None
        else [axis]
        if isinstance(axis, int)
        else list(axis)
    )

    # non-scaled
    if p == torch.inf:
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -torch.inf:
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return (x != 0).sum(dim=dim, keepdim=keepdim)

    # NOTE: preconditioning improves numerical stability
    x_max = x.amax(dim=dim, keepdim=True)
    result = x_max * (x / x_max).pow(p).sum(dim=dim, keepdim=True).pow(1 / p)
    return result.squeeze(dim=dim * (1 - int(keepdim)))  # branchless


@torch.compile(fullgraph=True)
def tensor_norm(
    x: Tensor,
    /,
    *,
    p: float = 2.0,
    axis: None | int | list[int] = None,
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:
    r"""Entry-wise norm of $p$-th order.

    .. signature:: ``(..., n) -> ...``

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


@torch.compile(fullgraph=True)
def multi_norm(
    tensors: list[Tensor],
    /,
    *,
    p: float = 2,
    q: float = 2,
    scaled: bool = True,
) -> Tensor:
    r"""Return the (scaled) p-q norm of the gradients.

    .. signature:: ``(...) -> ()``

    .. math:: ‖A‖_{p,q} ≔ \Bigl|∑_{j=1}^n \Big(∑_{i=1}^m |A_{ij}|^p\Big)^{q/p}\Bigr|^{1/q}

    If `normalize=True`, the sums are replaced with averages.
    """
    if not tensors:
        raise ValueError(
            "Input list of tensors is empty. Please provide at least one tensor."
        )

    first = tensors[0]

    # filter empty tensors
    tensors = [t for t in tensors if t.numel() > 0]

    if not tensors:
        return first.new_zeros(())

    s = tensor_norm(tensors[0], p=p, scaled=scaled) ** q
    for x in tensors[1:]:
        s += tensor_norm(x, p=p, scaled=scaled) ** q
    return (s / (1 + int(scaled) * len(tensors))) ** (1 / q)


@torch.compile(fullgraph=True)
def grad_norm(
    tensors: list[Tensor],
    /,
    *,
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
    if x.grad is None:
        raise ValueError("No gradients found for the first tensor.")
    s = tensor_norm(x.grad, p=p, scaled=scaled) ** q

    # iteration 2...n
    for x in tensors[1:]:
        if x.grad is None:
            raise ValueError("No gradients found for the first tensor.")
        s += tensor_norm(x.grad, p=p, scaled=scaled) ** q
    return (s / (1 + int(scaled) * len(tensors))) ** (1 / q)
