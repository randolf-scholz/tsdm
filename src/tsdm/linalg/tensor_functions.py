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
    r"""Compute the geometric mean along `dim`, ignoring NaNs in log space.

    With `dim=None`, reduce over every dimension. Negative real entries become
    NaN under the logarithm and are omitted; zeros contribute negative infinity
    in log space. A reduction with no valid log values produces NaN.
    """
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
    r"""Compute the size-normalized entry-wise norm; see `tensor_norm`."""
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
    r"""Compute the unscaled entry-wise norm; see `tensor_norm`.

    This helper is also exposed separately for testing.
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
    r"""Compute the entry-wise norm of order $p$ along `dim`.

    For finite nonzero $p$, the unscaled norm is $(∑ᵢ |xᵢ|ᵖ)^{1/p}$.
    Setting `scaled=True` replaces the sum with a mean.

    | Order  | Unscaled                                   | Size normalized                             |
    |--------|--------------------------------------------|---------------------------------------------|
    | $p=+∞$ | Maximum absolute value.                    | Maximum absolute value.                     |
    | $p=2$  | Root sum of squares.                       | Root mean square.                           |
    | $p=1$  | Sum of absolute values.                    | Mean absolute value.                        |
    | $p=0$  | Number of nonzero entries.                 | Geometric mean of absolute values.          |
    | $p=-1$ | Reciprocal sum of reciprocals.             | Harmonic mean of absolute values.           |
    | $p=-2$ | Reciprocal root sum of reciprocal squares. | Reciprocal root mean of reciprocal squares. |
    | $p=-∞$ | Minimum absolute value.                    | Minimum absolute value.                     |

    All reciprocal expressions use absolute values. With `dim=None`, reduce
    over every dimension. With `keepdim=True`, retain reduced axes as singleton
    dimensions. For finite nonzero orders, an all-zero reduction produces NaN
    because the implementation divides by its maximum absolute value.
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
    r"""Aggregate entry-wise norms of a nonempty list of tensors.

    With `scaled=False`, the aggregate for finite nonzero orders is

    $$ \Bigl(∑ⱼ₌₁ⁿ \Bigl(∑ᵢ |Aᵢⱼ|ᵖ\Bigr)^{q/p}\Bigr)^{1/q} $$

    Here $Aᵢⱼ$ is entry $i$ of tensor $j$; tensors may have different shapes.
    Empty tensors are omitted. With `scaled=True`, inner norms use means and
    the outer sum is divided by $n+1$, where $n$ counts nonempty tensors.

    Args:
        tensors: Tensors whose norms are aggregated.
        p: The inner norm order, passed to `tensor_norm`.
        q: The nonzero outer aggregation order.
        scaled: Whether to use the normalization described above.

    An empty list raises `ValueError`. A list containing only empty tensors
    returns a scalar zero with the first tensor's dtype and device.
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
    r"""Aggregate entry-wise norms of the gradients attached to tensors.

    With `scaled=False`, the aggregate for finite nonzero orders is

    $$ \Bigl(∑ⱼ₌₁ⁿ \Bigl(∑ᵢ |Gᵢⱼ|ᵖ\Bigr)^{q/p}\Bigr)^{1/q} $$

    Here $Gᵢⱼ$ is entry $i$ of the gradient of tensor $j$.
    With `scaled=True`, inner norms use means and the outer sum is divided by
    $n+1$, where $n$ is the number of input tensors.

    Args:
        tensors: Tensors with populated `.grad` attributes.
        p: The inner norm order, passed to `tensor_norm`.
        q: The nonzero outer aggregation order.
        scaled: Whether to use the normalization described above.

    An empty list returns a scalar zero with PyTorch's default dtype and device.
    A missing gradient on any input tensor raises `ValueError`.
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
