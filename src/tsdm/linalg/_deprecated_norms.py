r"""TODO: Module Docstring.

TODO: Module summary.
"""

__all__ = [
    # Constants
    "SizeLike",
    # Functions
    "relative_error",
    "scaled_norm",
    "multi_scaled_norm",
]

from collections.abc import Iterable, Sequence
from functools import singledispatch
from typing import Optional, TypeAlias, cast, overload

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

SizeLike: TypeAlias = int | tuple[int, ...]
r"""Type hint for a size-like object."""


@singledispatch
def relative_error(
    xhat: ArrayLike | Tensor, x_true: ArrayLike | Tensor, /
) -> NDArray | Tensor:
    r"""Relative error, works with both `Tensor` and `ndarray`.

    .. math:: r(x̂, x) = \tfrac{|x̂ - x|}{|x|+ε}

    The tolerance parameter $ε$ is determined automatically. By default,
    $ε=2^{-24}$ for single and $ε=2^{-53}$ for double precision.
    """
    xhat, x_true = np.asanyarray(xhat), np.asanyarray(x_true)
    return _numpy_relative_error(xhat, x_true)


@relative_error.register
def _numpy_relative_error(xhat: np.ndarray, x_true: np.ndarray, /) -> np.ndarray:
    if xhat.dtype in (np.float16, np.int16):
        eps = 2**-11
    elif xhat.dtype in (np.float32, np.int32):
        eps = 2**-24
    elif xhat.dtype in (np.float64, np.int64):
        eps = 2**-53
    else:
        raise NotImplementedError

    return np.abs(xhat - x_true) / (np.abs(x_true) + eps)


@relative_error.register
def _torch_relative_error(xhat: Tensor, x_true: Tensor, /) -> Tensor:
    if xhat.dtype in (torch.bfloat16,):
        eps = 2**-8
    elif xhat.dtype in (torch.float16, torch.int16):
        eps = 2**-11
    elif xhat.dtype in (torch.float32, torch.int32):
        eps = 2**-24
    elif xhat.dtype in (torch.float64, torch.int64):
        eps = 2**-53
    else:
        raise NotImplementedError

    # eps = eps or _eps
    return torch.abs(xhat - x_true) / (torch.abs(x_true) + eps)


@overload
def scaled_norm(
    x: Tensor,
    /,
    *,
    p: float = 2,
    axis: Optional[SizeLike] = None,
    keepdims: bool = False,
) -> Tensor: ...
@overload
def scaled_norm(
    x: NDArray,
    /,
    *,
    p: float = 2,
    axis: Optional[SizeLike] = None,
    keepdims: bool = False,
) -> NDArray: ...
@overload
def scaled_norm(
    x: Sequence[Tensor],
    /,
    *,
    p: float = 2,
    axis: Optional[SizeLike] = None,
    keepdims: bool = False,
) -> Tensor: ...
@overload
def scaled_norm(
    x: Sequence[NDArray],
    /,
    *,
    p: float = 2,
    axis: Optional[SizeLike] = None,
    keepdims: bool = False,
) -> NDArray: ...
def scaled_norm(
    x: Tensor | NDArray | Sequence[Tensor] | Sequence[NDArray],
    /,
    *,
    p: float = 2,
    axis: Optional[SizeLike] = None,
    keepdims: bool = False,
) -> Tensor | NDArray:
    r"""Scaled $ℓ^p$-norm, works with both `Tensor` and `ndarray`.

    .. math:: ‖x‖_p = (⅟ₙ ∑_{i=1}^n |x_i|^p)^{1/p}

    This naturally leads to

    .. math:: ∥u⊕v∥ = \frac{\dim U}{\dim U⊕V} ∥u∥ + \frac{\dim V}{\dim U⊕V} ∥v∥

    .. math:: ∥u⊕v∥_p^p = \frac{\dim U}{\dim U⊕V} ∥u∥_p^p + \frac{\dim V}{\dim U⊕V} ∥v∥_p^p

    This choice is consistent with associativity: $∥(u⊕v)⊕w∥ = ∥u⊕(v⊕w)∥$

    In particular, given $𝓤=⨁_{i=1:n} U_i$, then

    .. math:: ∥u∥_p^p = ∑_{i=1:n} \frac{\dim U_i}{\dim 𝓤} ∥u_i∥_p^p
    """
    if isinstance(x, Tensor):
        axis = () if axis is None else axis
        return _torch_scaled_norm(x, p=p, axis=axis, keepdims=keepdims)
    if isinstance(x, np.ndarray):
        return _numpy_scaled_norm(x, p=p, axis=axis, keepdims=keepdims)
    if isinstance(x[0], Tensor):
        x = cast(Sequence[Tensor], x)
        return _torch_multi_scaled_norm(x, p=p, q=p)
    x = cast(Sequence[NDArray], x)
    return _numpy_multi_scaled_norm(x, p=p, q=p)


def _torch_scaled_norm(
    x: Tensor,
    /,
    *,
    p: float = 2,
    axis: SizeLike = (),  # FIXME: use tuple[int, ...] https://github.com/pytorch/pytorch/issues/64700
    keepdims: bool = False,
) -> Tensor:
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)
    x = torch.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return torch.exp(torch.mean(torch.log(x), dim=axis, keepdim=keepdims))
    if p == 1:
        return torch.mean(x, dim=axis, keepdim=keepdims)
    if p == 2:
        return torch.sqrt(torch.mean(x**2, dim=axis, keepdim=keepdims))
    if p == float("inf"):
        return torch.amax(x, dim=axis, keepdim=keepdims)
    # other p
    return torch.mean(x**p, dim=axis, keepdim=keepdims) ** (1 / p)


def _numpy_scaled_norm(
    x: NDArray,
    /,
    *,
    p: float = 2,
    axis: Optional[SizeLike] = None,
    keepdims: bool = False,
) -> NDArray:
    x = np.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return np.exp(np.mean(np.log(x), axis=axis, keepdims=keepdims))
    if p == 1:
        return np.mean(x, axis=axis, keepdims=keepdims)
    if p == 2:
        return np.sqrt(np.mean(x**2, axis=axis, keepdims=keepdims))
    if p == float("inf"):
        return np.max(x, axis=axis, keepdims=keepdims)
    # other p
    return np.mean(x**p, axis=axis, keepdims=keepdims) ** (1 / p)


@overload
def multi_scaled_norm(
    x: Sequence[Tensor],
    /,
    *,
    p: float = 2,
) -> Tensor: ...
@overload
def multi_scaled_norm(
    x: Sequence[NDArray],
    /,
    *,
    p: float = 2,
) -> NDArray: ...
def multi_scaled_norm(
    x: Sequence[Tensor] | Sequence[NDArray],
    /,
    *,
    p: float = 2,
    q: float = 2,
) -> Tensor | NDArray:
    # TODO: figure out correct normalization
    r"""Scaled $L_{p,q}$-norm.

    .. math::
        ∥u_1⊕…⊕u_n∥_{\operatorname*{⨁}_{i=1:n}U_i}
          &≔ ∥v∥_q \qq{where} v_i = ∥u_i∥_p
        \\&= ∑_{i=1:n} \frac{\dim U_i}{\dim 𝓤} ∥u_i∥_p
        \\&= \Bigl(\frac{1}{n} ∑_{i=1:n}
                \Bigr(\frac{1}{m_i}∑_{j=1:m_i} |(u_i)_j|^{p}\Bigl)^{q/p}
             \Bigr)^{1/q}
    """
    if isinstance(x[0], Tensor):
        x = cast(Sequence[Tensor], x)
        return _torch_multi_scaled_norm(x, p=p, q=q)
    x = cast(Sequence[NDArray], x)
    return _numpy_multi_scaled_norm((np.asarray(z) for z in x), p=p, q=q)


def _torch_multi_scaled_norm(
    x: Iterable[Tensor],
    /,
    *,
    p: float = 2,
    q: float = 2,
) -> Tensor:
    z = torch.stack([_torch_scaled_norm(z, p=p) ** q for z in x])
    w = torch.tensor([z.numel() for z in x], device=z.device, dtype=z.dtype)
    return (torch.dot(w, z) / torch.sum(w)) ** (1 / q)


def _numpy_multi_scaled_norm(
    x: Iterable[NDArray],
    /,
    *,
    p: float = 2,
    q: float = 2,
) -> NDArray:
    z = np.stack([_numpy_scaled_norm(z, p=p) ** q for z in x])
    w = np.array([z.size for z in x])
    return (np.dot(w, z) / np.sum(w)) ** (1 / q)
