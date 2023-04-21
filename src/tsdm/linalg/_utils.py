r"""Utility Linear Algebra Functions."""

__all__ = [
    # Functions
    "apply_along_axes",
]

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from tsdm.types.variables import tensor_var


def apply_along_axes(
    a: tensor_var, b: tensor_var, op: Callable, axes: tuple[int, ...]
) -> tensor_var:
    r"""Apply a function to multiple axes of a tensor."""
    axes = tuple(axes)
    rank = len(a.shape)
    source = tuple(range(rank))
    inverse_permutation: tuple[int, ...] = axes + tuple(
        ax for ax in range(rank) if ax not in axes
    )
    perm = tuple(np.argsort(inverse_permutation))
    if isinstance(a, Tensor):
        a = torch.moveaxis(a, source, perm)
        a = op(a, b)
        a = torch.moveaxis(a, source, inverse_permutation)
    else:
        a = np.moveaxis(a, source, perm)
        a = op(a, b)
        a = np.moveaxis(a, source, inverse_permutation)
    return a
