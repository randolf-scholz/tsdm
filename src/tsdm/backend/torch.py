"""Implementations for torch backend."""

__all__ = [
    "torch_nanmin",
    "torch_nanmax",
    "torch_nanstd",
]

import torch
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
