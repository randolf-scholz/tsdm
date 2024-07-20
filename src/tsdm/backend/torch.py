r"""Implementations for torch backend."""

__all__ = [
    # Functions
    "apply_along_axes",
    "copy_like",
    "drop_null",
    "nanmax",
    "nanmin",
    "nanstd",
    "scalar",
]

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from tsdm.types.aliases import Axis


def scalar(x: Any, /, dtype: Any) -> Any:
    return torch.tensor(x, dtype=dtype).item()


def drop_null(x: Tensor, /) -> Tensor:
    r"""Drop `NaN` values from a tensor, flattening it."""
    return x[~torch.isnan(x)].flatten()


def nanmin(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanmin`."""
    return torch.amin(
        torch.where(torch.isnan(x), float("+inf"), x),
        dim=axis,  # type: ignore[arg-type]
        keepdim=keepdims,
    )


def nanmax(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanmax`."""
    return torch.amax(
        torch.where(torch.isnan(x), float("-inf"), x),
        dim=axis,  # type: ignore[arg-type]
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


def apply_along_axes(
    op: Callable[..., Tensor], /, *tensors: Tensor, axis: Axis
) -> Tensor:
    r"""Apply a function to multiple tensors along axes.

    It is assumed that all tensors have the same shape.
    """
    if len(tensors) < 1:
        raise ValueError("At least one tensor is required!")
    if len({t.shape for t in tensors}) != 1:
        raise ValueError("All tensors must have the same shape!")

    axes = () if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)

    rank = len(tensors[0].shape)
    source = tuple(range(rank))
    inverse_permutation = axes + tuple(ax for ax in range(rank) if ax not in axes)
    perm = tuple(np.argsort(inverse_permutation))
    tensors = tuple(torch.moveaxis(tensor, source, perm) for tensor in tensors)
    result = op(*tensors)
    result = torch.moveaxis(result, source, inverse_permutation)
    return result
