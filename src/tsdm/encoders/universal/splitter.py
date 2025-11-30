r"""Encoders for numpy arrays, can also be applied to pandas dataframes.

Numerical Encoders should be able to be applied with different backends such as

- numpy arrays
- pandas dataframes
- torch tensors
- pyarrow tables
- etc.

To ensure performance during encoding/decoding, the backend should be fixed.

Goals
=====
- numerical encoders should allow for different backends: numpy, pandas, torch, etc.
- numerical encoders should be vectorized and fast
- we should be able to "slice" vectorized encoders just like we slice numpy arrays
- calling fit twice on the same data should not change the encoder (idempotent)
- one should be able to (partially or fully) fix the encoder parameters
- should have an axis attribute that allows for broadcasting.
- switching between backends should be easy and fast
    - switching between backends probably not considered a "fit" operation
    - fitting changes the encoder parameter values, switching backends changes their types.
"""

__all__ = [
    "TensorConcatenator",
    "TensorSplitter",
]

from dataclasses import KW_ONLY, dataclass

from tsdm.backend import Backend, get_backend
from tsdm.backend.types import NumericalArray as Array
from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder
from tsdm.utils.decorators import pprint_repr


@pprint_repr
@dataclass
class TensorSplitter[Arr: Array](FittableEncoder[Arr, list[Arr]]):
    r"""Split tensor along specified axis."""

    _: KW_ONLY
    indices: int | list[int] = 1
    axis: int = 0
    backend: Backend = UNDEFINED

    def __invert__(self) -> "TensorConcatenator[Arr]":
        return TensorConcatenator(
            axis=self.axis, indices=self.indices, backend=self.backend
        )

    def fit(self, x: Arr, /) -> None:
        self.backend = get_backend(x)

    def encode(self, x: Arr, /) -> list[Arr]:
        return self.backend.array_split(x, self.indices, axis=self.axis)

    def decode(self, y: list[Arr], /) -> Arr:
        return self.backend.concatenate(y, axis=self.axis)


@pprint_repr
@dataclass
class TensorConcatenator[Arr: Array](FittableEncoder[list[Arr], Arr]):
    r"""Concatenate multiple tensors."""

    _: KW_ONLY
    indices: int | list[int] = UNDEFINED
    axis: int = 0
    backend: Backend = UNDEFINED

    def __invert__(self) -> TensorSplitter[Arr]:
        return TensorSplitter(
            axis=self.axis, indices=self.indices, backend=self.backend
        )

    def fit(self, x: list[Arr], /) -> None:
        self.backend = get_backend(x)
        self.indices = [arr.shape[self.axis] for arr in x]

    def encode(self, x: list[Arr], /) -> Arr:
        return self.backend.concatenate(x, axis=self.axis)

    def decode(self, y: Arr, /) -> list[Arr]:
        return self.backend.array_split(y, self.indices, axis=self.axis)
