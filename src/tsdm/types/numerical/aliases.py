r"""Alias types for numerical computations."""

__all__ = [
    "PythonScalar",
    "Axis",
    "Dims",
    "Size",
    "Shape",
]

from datetime import datetime, timedelta

# region Scalar Type Aliases -----------------------------------------------------------
type PythonScalar = bool | int | float | complex | str | bytes | datetime | timedelta
r"""Type Alias for Python scalars."""
# endregion Scalar Type Aliases --------------------------------------------------------

type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Dims = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""  # FIXME: https://github.com/pytorch/pytorch/issues/64700
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type Shape = int | tuple[int, ...]
r"""Type Alias for shape-like objects (note: `ones(shape=None)` creates 0d-array."""
