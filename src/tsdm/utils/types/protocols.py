r"""Protocols used in tsdm.

References
----------
- https://www.python.org/dev/peps/pep-0544/
- https://docs.python.org/3/library/typing.html#typing.Protocol
- https://numpy.org/doc/stable/reference/c-api/array.html
"""

from __future__ import annotations

__all__ = [
    # Classes
    "Array",
    "NTuple",
    "Dataclass",
]

from collections.abc import Iterable, Sequence
from dataclasses import Field
from typing import Protocol, TypeAlias, TypeVar, Union, overload, runtime_checkable

ScalarType = TypeVar("ScalarType")
T = TypeVar("T")
Either: TypeAlias = Union[T, "Array[T]"]
Self = TypeVar("Self", bound="Array")


@runtime_checkable
class Dataclass(Protocol):
    r"""Protocol for anonymous dataclasses."""

    __dataclass_fields__: dict[str, Field]


@runtime_checkable
class NTuple(Protocol):
    r"""Protocol for anonymous namedtuple."""

    @property
    def _fields(self) -> tuple[str, ...]:
        return ()

    def _asdict(self) -> dict[str, object]:
        ...


@runtime_checkable
class Array(Protocol[ScalarType]):
    r"""We just test for shape, since e.g. tf.Tensor does not have ndim."""

    @property
    def shape(self) -> Iterable[int]:
        r"""Yield the shape of the array."""

    @property
    def dtype(self) -> type[ScalarType]:
        r"""Yield the data type of the array."""

    # fmt: off
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> ScalarType: ...
    @overload
    def __getitem__(self, key: Sequence[bool]) -> Array[ScalarType]: ...
    @overload
    def __getitem__(self, key: Sequence[int]) -> Array[ScalarType]: ...
    def __getitem__(self, key): ...
    def __eq__(self, other: Either[ScalarType]) -> Array[bool]: ...  # type: ignore[override]
    def __le__(self, other: Either[ScalarType]) -> Array[bool]: ...
    def __ge__(self, other: Either[ScalarType]) -> Array[bool]: ...
    def __lt__(self, other: Either[ScalarType]) -> Array[bool]: ...
    def __gt__(self, other: Either[ScalarType]) -> Array[bool]: ...
    def __ne__(self, other: Either[ScalarType]) -> Array[bool]: ...  # type: ignore[override]
    def __neg__(self) -> Array[ScalarType]: ...
    def __invert__(self) -> Array[ScalarType]: ...
    def __add__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __radd__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __iadd__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __sub__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __rsub__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __isub__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __mul__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __rmul__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __imul__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __truediv__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __rtruediv__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    def __itruediv__(self, other: Either[ScalarType]) -> Array[ScalarType]: ...
    # fmt: on
