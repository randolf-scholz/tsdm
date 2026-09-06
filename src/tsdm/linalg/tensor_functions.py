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

type Dim = None | int | tuple[int, ...]


def geometric_mean(
    x: Tensor,  # Float[..., *D]
    /,
    *,
    dim: Dim = None,  # *D
    keepdim: bool = False,
) -> Tensor:  # Float[...]
    r"""Geometric mean of a tensor."""
    dim = (
        tuple(range(x.ndim))
        if dim is None
        else (dim,)
        if isinstance(dim, int)
        else tuple(dim)
    )

    return x.log().nanmean(dim=dim, keepdim=keepdim).exp()


def scaled_norm(
    x: Tensor,  # Float[..., *D]
    /,
    *,
    p: float = 2.0,
    dim: Dim = None,
    keepdim: bool = False,
) -> Tensor:  # Float[...]
    r"""Shortcut for scaled norm."""
    # TODO: deal with nan values
    dim = (
        tuple(range(x.ndim))
        if dim is None
        else (dim,)
        if isinstance(dim, int)
        else tuple(dim)
    )
    x = x.abs()

    if p == torch.inf:
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -torch.inf:
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        return geometric_mean(x, dim=dim, keepdim=keepdim)

    # NOTE: preconditioning with x_max is not necessary, but it helps with numerical stability and prevents overflow
    x_max = x.abs().amax(dim=dim, keepdim=True)
    result = x_max * (x / x_max).pow(p).mean(dim=dim, keepdim=True).pow(1 / p)
    return result.squeeze(dim=dim * (1 - int(keepdim)))  # branchless


def norm(
    x: Tensor,  # Float[..., *D]
    /,
    *,
    p: float = 2.0,
    dim: Dim = None,
    keepdim: bool = False,
) -> Tensor:  # Float[...]
    r"""Shortcut for non-scaled norm.

    Only present here for testing purposes.
    """
    # TODO: deal with nan values
    x = x.abs()

    dim = (
        tuple(range(x.ndim))
        if dim is None
        else (dim,)
        if isinstance(dim, int)
        else tuple(dim)
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


def tensor_norm(
    x: Tensor,  # Float[..., *D]
    /,
    *,
    p: float = 2.0,
    dim: Dim = None,
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:  # Float[...]
    r"""Entry-wise norm of $p$-th order.

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
        scaled_norm(x, p=p, dim=dim, keepdim=keepdim)
        if scaled
        else norm(x, p=p, dim=dim, keepdim=keepdim)
    )


def multi_norm(
    tensors: list[Tensor],  # list[Float[...]]
    /,
    *,
    p: float = 2,
    q: float = 2,
    scaled: bool = True,
) -> Tensor:  # Float[()]
    r"""Return the (scaled) p-q norm of the gradients.

    .. math:: ‖A‖_{p,q} ≔ \Bigl|∑ⱼ₌₁ⁿ \Big(∑ᵢ₌₁ᵐ |Aᵢⱼ|ᵖ\Big)^{q/p}\Bigr|^{1/q}

    Args:
        tensors: `list[Float[...]]`
        p: the inner norm order
        q: the outer norm order
        scaled: If `True`, the sums are replaced with averages.
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


def grad_norm(
    tensors: list[Tensor],  # list[Float[...]]
    /,
    *,
    p: float = 2,
    q: float = 2,
    scaled: bool = True,
) -> Tensor:  # Float[()]
    r"""Return the (scaled) p-q norm of the gradients.

    .. math:: ‖A‖_{p,q} ≔ \Bigl|∑ⱼ₌₁ⁿ \Big(∑ᵢ₌₁ᵐ |Aᵢⱼ|ᵖ\Big)^{q/p}\Bigr|^{1/q}

    Args:
        tensors: `list[Float[...]]`
        p: the inner norm order
        q: the outer norm order
        scaled: If `True`, the sums are replaced with averages.
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
