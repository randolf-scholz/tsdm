"""Implementations for torch backend."""

__all__ = [
    "torch_nanmin",
    "torch_nanmax",
    "torch_nanstd",
    "torch_like",
]

import torch
from numpy.typing import ArrayLike
from torch import Tensor

from tsdm.types.aliases import Axes


def torch_nanmin(x: Tensor, /, *, axis: Axes = None, keepdims: bool = False) -> Tensor:
    """Analogue to `numpy.nanmin`."""
    return torch.amin(
        torch.where(torch.isnan(x), float("+inf"), x),
        dim=axis,  # type: ignore[arg-type]
        keepdim=keepdims,
    )


def torch_nanmax(x: Tensor, /, *, axis: Axes = None, keepdims: bool = False) -> Tensor:
    """Analogue to `numpy.nanmax`."""
    return torch.amax(
        torch.where(torch.isnan(x), float("-inf"), x),
        dim=axis,  # type: ignore[arg-type]
        keepdim=keepdims,
    )


def torch_nanstd(x: Tensor, /, *, axis: Axes = None, keepdims: bool = False) -> Tensor:
    """Analogue to `numpy.nanstd`."""
    return torch.sqrt(
        torch.nanmean(  # type: ignore[call-arg]
            (x - torch.nanmean(x, dim=axis, keepdim=True)) ** 2,
            axis=axis,
            keepdim=keepdims,
        )
    )


def torch_like(x: ArrayLike, ref: Tensor, /) -> Tensor:
    """Return a tensor of the same dtype and other options as `ref`."""
    return torch.tensor(x, dtype=ref.dtype, device=ref.device)
