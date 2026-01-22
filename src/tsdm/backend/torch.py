r"""Implementations for torch backend."""

__all__ = [
    # Constants
    "EPS",
    # Functions
    "apply_along_axes",
    "copy_like",
    "drop_null",
    "nanmax",
    "nanmin",
    "nanstd",
    "scalar",
    # utils
]

from collections.abc import Callable as Fn
from typing import Any, Final

import torch
from numpy.typing import ArrayLike
from torch import Tensor

from tsdm.types.aliases import Axis

EPS: Final[dict[torch.dtype, float]] = {
    torch.bfloat16   : 1e-2,
    torch.complex128 : 1e-15,
    torch.complex32  : 1e-3,
    torch.complex64  : 1e-6,
    torch.float16    : 1e-3,
    torch.float32    : 1e-6,
    torch.float64    : 1e-15,
}  # fmt: skip
r"""CONST: Default epsilon for each dtype."""


def scalar(x: Any, /, dtype: Any) -> Any:
    return torch.tensor(x, dtype=dtype).item()


def drop_null(x: Tensor, /) -> Tensor:
    r"""Drop `NaN` values from a tensor, flattening it."""
    return x[~torch.isnan(x)].flatten()


def nanmin(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanmin`."""
    return torch.amin(
        torch.where(torch.isnan(x), float("+inf"), x),
        dim=axis,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        keepdim=keepdims,
    )


def nanmax(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanmax`."""
    return torch.amax(
        torch.where(torch.isnan(x), float("-inf"), x),
        dim=axis,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        keepdim=keepdims,
    )


def nanstd(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanstd`."""
    r = x - torch.nanmean(x, dim=axis, keepdim=True)
    return torch.sqrt(
        torch.nanmean(
            r.pow(2),
            dim=axis,
            keepdim=keepdims,
        )
    )


def copy_like(x: ArrayLike, ref: Tensor, /) -> Tensor:
    r"""Return a tensor of the same dtype and other options as `ref`."""
    return torch.tensor(x, dtype=ref.dtype, device=ref.device)


def apply_along_axes(op: Fn[..., Tensor], /, *tensors: Tensor, axis: Axis) -> Tensor:
    r"""Apply a function to multiple tensors along axes.

    Assumptions:
    - All tensors must have the same shape.
    - The operator `op` acts on the last `len(axis)` axes of the tensors.
    - The operator `op` does not change the shape of the tensors.
    """
    if len(tensors) < 1:
        raise ValueError("At least one tensor is required!")
    if len({t.shape for t in tensors}) != 1:
        raise ValueError("All tensors must have the same shape!")

    # we move the target axes to the front, apply the operation,
    # then move them back to their original position
    rank = len(tensors[0].shape)
    target_axes = (
        () if axis is None
        else (axis % rank,) if isinstance(axis, int)
        else tuple(ax % rank for ax in axis)
    )  # fmt: skip
    other_axes = tuple(ax for ax in range(rank) if ax not in target_axes)
    source = tuple(range(rank))
    inv = target_axes + other_axes  # inverse permutation
    dest = tuple(sorted(source, key=inv.__getitem__))
    tensors = tuple(torch.moveaxis(tensor, source, dest) for tensor in tensors)
    result = op(*tensors)
    result = torch.moveaxis(result, source, dest)
    return result
