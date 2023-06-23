"""Numerical functions with shared signatures for multiple backends."""

__all__ = ["Backend", "get_backend", "where_kernel", "clip_kernel"]

from typing import Any, Callable, Literal, TypeAlias, TypeVar

import numpy
import torch
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Series
from torch import Tensor

T = TypeVar("T", Series, DataFrame, Tensor, ndarray)
Backend: TypeAlias = Literal["torch", "numpy", "pandas"]


def get_backend(*objs: Any, fallback: Backend = "numpy") -> Backend:
    """Get the backend of a set of objects."""
    types: set[Backend] = set()
    for obj in objs:
        match obj:
            case Tensor():
                types.add("torch")
            case DataFrame() | Series():  #  type: ignore[misc]
                types.add("pandas")  # type: ignore[unreachable]
            case ndarray():
                types.add("numpy")
            case _:
                pass

    match len(types):
        case 0:
            return fallback
        case 1:
            return types.pop()
        case _:
            raise ValueError(f"More than 1 backend detected: {types}.")


where_kernel: dict[Backend, Callable[[T, T, T], T]] = {
    "torch": torch.where,
    "pandas": lambda cond, a, b: a.where(cond, b),
    "numpy": numpy.where,
}

clip_kernel: dict[Backend, Callable[[T, T | None, T | None], T]] = {
    "torch": torch.clip,
    "pandas": lambda x, lower=None, upper=None: x.clip(lower, upper),
    "numpy": numpy.clip,
}
