r"""Protocols used in tsdm.

References
----------
- https://www.python.org/dev/peps/pep-0544/
- https://docs.python.org/3/library/typing.html#typing.Protocol
- https://numpy.org/doc/stable/reference/c-api/array.html
"""

__all__ = [
    # Classes
    "Array",
    "Dataclass",
    "Hash",
    "Lookup",
    "NTuple",
]

from collections.abc import Sequence
from dataclasses import Field
from typing import Any, Protocol, TypeVar, runtime_checkable

from tsdm.types.variables import Key_contra, KeyVar, Value_co

ScalarType_co = TypeVar("ScalarType_co", covariant=True)
# Either: TypeAlias = Union[T, "Array[T]"]
A = TypeVar("A", bound="Array")


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
class Array(Protocol[ScalarType_co]):  # FIXME: Use Self
    r"""We just test for shape, since e.g. tf.Tensor does not have ndim."""

    @property
    def shape(self) -> Sequence[int]:
        r"""Yield the shape of the array."""

    @property
    def ndim(self) -> int:
        r"""Number of dimensions."""

    @property
    def dtype(self) -> ScalarType_co:
        r"""Yield the data type of the array."""

    # fmt: off
    def __len__(self) -> int: ...
    # @overload
    # def __getitem__(self: A, key: int) -> Any: ...
    # @overload
    # def __getitem__(self: A, key: Sequence[bool]) -> A: ...
    # @overload
    # def __getitem__(self: A, key: Sequence[int]) -> A: ...
    def __getitem__(self: A, key: Any) -> A: ...
    def __eq__(self: A, other: Any) -> A: ...  # type: ignore[override]
    def __le__(self: A, other: Any) -> A: ...
    def __ge__(self: A, other: Any) -> A: ...
    def __lt__(self: A, other: Any) -> A: ...
    def __gt__(self: A, other: Any) -> A: ...
    def __ne__(self: A, other: Any) -> A: ...  # type: ignore[override]
    def __neg__(self: A) -> A: ...
    def __invert__(self: A) -> A: ...
    def __add__(self: A, other: Any) -> A: ...
    def __radd__(self: A, other: Any) -> A: ...
    def __iadd__(self: A, other: Any) -> A: ...
    def __sub__(self: A, other: Any) -> A: ...
    def __rsub__(self: A, other: Any) -> A: ...
    def __isub__(self: A, other: Any) -> A: ...
    def __mul__(self: A, other: Any) -> A: ...
    def __rmul__(self: A, other: Any) -> A: ...
    def __imul__(self: A, other: Any) -> A: ...
    def __truediv__(self: A, other: Any) -> A: ...
    def __rtruediv__(self: A, other: Any) -> A: ...
    def __itruediv__(self: A, other: Any) -> A: ...
    # fmt: on


@runtime_checkable
class Lookup(Protocol[Key_contra, Value_co]):
    """Mapping/Sequence like generic that is contravariant in Keys."""

    def __contains__(self, key: KeyVar) -> bool:
        # Here, any Hashable input is accepted.
        """Return True if the map contains the given key."""

    def __getitem__(self, key: Key_contra) -> Value_co:
        """Return the value associated with the given key."""

    def __len__(self) -> int:
        """Return the number of items in the map."""


@runtime_checkable
class Hash(Protocol):
    """Protocol for hash-functions."""

    name: str

    def digest_size(self) -> int:
        """Return the size of the hash in bytes."""

    def block_size(self) -> int:
        """Return the internal block size of the hash in bytes."""

    def update(self, data: bytes) -> None:
        """Update this hash object's state with the provided string."""

    def digest(self) -> bytes:
        """Return the digest value as a string of binary data."""

    def hexdigest(self) -> str:
        """Return the digest value as a string of hexadecimal digits."""

    def copy(self) -> Any:  # TODO: Use typing.Self
        """Return a clone of the hash object."""
