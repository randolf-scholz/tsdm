r"""PyTorch backend implementations for tsdm."""

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
r"""Default epsilon for each supported dtype."""


def scalar(x: Any, /, dtype: Any) -> Any:
    return torch.tensor(x, dtype=dtype).item()


def drop_null(x: Tensor, /) -> Tensor:
    r"""Drop `NaN` values from a tensor, flattening it."""
    return x[~torch.isnan(x)].flatten()


def nanmin(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Return the minimum along the selected axes, ignoring `NaN`.

    The input is assumed to be a floating-point tensor.
    An all-NaN slice produces `NaN` and emits a `RuntimeWarning`, as in NumPy.
    """
    assert x.is_floating_point(), "Expected a floating-point tensor."
    if axis == ():
        # Reduce an added singleton axis to preserve NumPy's elementwise behavior.
        # XREF: https://github.com/pytorch/pytorch/issues/29137.
        x, axis, keepdims = x.unsqueeze(-1), -1, False
    mask = x.isnan()
    # FIXME: https://github.com/pytorch/pytorch/issues/156072
    result = x.masked_fill(mask, torch.inf).amin(dim=axis, keepdim=keepdims)  # type: ignore
    all_nan = mask.all(dim=axis, keepdim=keepdims)
    return result.masked_fill(all_nan, torch.nan)


def nanmax(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Return the maximum along the selected axes, ignoring `NaN`.

    The input is assumed to be a floating-point tensor.
    An all-NaN slice produces `NaN` and emits a `RuntimeWarning`, as in NumPy.
    """
    assert x.is_floating_point(), "Expected a floating-point tensor."
    if axis == ():
        # Reduce an added singleton axis to preserve NumPy's elementwise behavior.
        # XREF: https://github.com/pytorch/pytorch/issues/29137.
        x, axis, keepdims = x.unsqueeze(-1), -1, False
    mask = x.isnan()
    # FIXME: https://github.com/pytorch/pytorch/issues/156072
    result = x.masked_fill(mask, -torch.inf).amax(dim=axis, keepdim=keepdims)  # type: ignore
    all_nan = mask.all(dim=axis, keepdim=keepdims)
    return result.masked_fill(all_nan, torch.nan)


def nanstd(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Return the population standard deviation along axes, ignoring `NaN`.

    For each slice, let $I$ index its non-NaN entries and $N = |I|$. Then

    $$ μ = \frac{1}{N} ∑_{i ∈ I} xᵢ, \qquad σ = \sqrt{\frac{1}{N} ∑_{i ∈ I} |xᵢ - μ|²} $$

    An empty or all-NaN slice produces `NaN` and emits a `RuntimeWarning`.
    The variance uses zero degrees of freedom correction, matching the default
    behavior of `numpy.nanstd`. Complex inputs produce real standard deviations.
    """
    assert x.is_floating_point() or x.is_complex(), (
        "Expected a floating-point or complex tensor."
    )
    if axis == ():
        # NumPy reduces no axes for axis=(); Torch reduces all axes for dim=().
        # Reduce an added singleton axis to preserve NumPy's elementwise behavior.
        # XREF: https://github.com/pytorch/pytorch/issues/29137.
        x, axis, keepdims = x.unsqueeze(-1), -1, False
    mask = x.isnan()
    count = (~mask).sum(dim=axis, keepdim=True)
    mean = x.masked_fill(mask, 0).sum(dim=axis, keepdim=True) / count
    r = (x - mean).masked_fill(mask, 0)
    count = (~mask).sum(dim=axis, keepdim=keepdims)
    return (r.abs().square().sum(dim=axis, keepdim=keepdims) / count).sqrt()


def copy_like(x: ArrayLike, ref: Tensor, /) -> Tensor:
    r"""Create a tensor from `x` using the dtype and device of `ref`."""
    return torch.tensor(x, dtype=ref.dtype, device=ref.device)


def apply_along_axes(op: Fn[..., Tensor], /, *tensors: Tensor, axis: Axis) -> Tensor:
    r"""Apply a function to multiple tensors along axes.

    Move the selected axes to the front, call `op`, and restore the original axis
    order. The operation must preserve the shape of its inputs.

    Args:
        op: Function applied to the tensors after moving the selected axes.
        *tensors: One or more tensors with identical shapes.
        axis: Axes to move to the front. Negative axes are normalized modulo the
            rank. `None` leaves the axis order unchanged.

    Raises:
        ValueError: If no tensors are supplied or their shapes differ.
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
