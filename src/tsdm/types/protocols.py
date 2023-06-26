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
    "SupportsShape",
    "Dataclass",
    "Hash",
    "Lookup",
    "NTuple",
]

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import Field
from typing import Any, Protocol, TypeVar, overload, runtime_checkable

from typing_extensions import Self, SupportsIndex

from tsdm.types.variables import any_co as T_co
from tsdm.types.variables import key_contra, key_var, value_co

scalar_co = TypeVar("scalar_co", covariant=True)
# Either: TypeAlias = Union[T, "Array[T]"]
A = TypeVar("A", bound="Array")


@runtime_checkable
class Dataclass(Protocol):
    r"""Protocol for anonymous dataclasses."""

    @property
    def __dataclass_fields__(self) -> Mapping[str, Field]:
        r"""Return the fields of the dataclass."""


@runtime_checkable
class NTuple(Protocol[T_co]):
    r"""Protocol for anonymous namedtuple.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
    """
    # fmt: off
    @property
    def _fields(self) -> tuple[str, ...]: ...
    def _asdict(self) -> Mapping[str, T_co]: ...
    # def __new__(cls, __iterable: Iterable[T_co] = ...) -> Self: ...
    def __len__(self) -> int: ...
    def __contains__(self, __key: object) -> bool: ...
    @overload
    def __getitem__(self, __key: SupportsIndex) -> T_co: ...
    @overload
    def __getitem__(self, __key: slice) -> tuple[T_co, ...]: ...
    def __iter__(self) -> Iterator[T_co]: ...
    # def __lt__(self, __value: tuple[T_co, ...]) -> bool: ...
    # def __le__(self, __value: tuple[T_co, ...]) -> bool: ...
    # def __gt__(self, __value: tuple[T_co, ...]) -> bool: ...
    # def __ge__(self, __value: tuple[T_co, ...]) -> bool: ...
    # @overload
    # def __add__(self, __value: tuple[T_co, ...]) -> tuple[T_co, ...]: ...
    # @overload
    # def __add__(self, __value: tuple[T, ...]) -> tuple[T_co | T, ...]: ...
    # def __mul__(self, __value: SupportsIndex) -> tuple[T_co, ...]: ...
    # def __rmul__(self, __value: SupportsIndex) -> tuple[T_co, ...]: ...
    # def count(self, __value: Any) -> int: ...
    # def index(self, __value: Any, __start: SupportsIndex = 0, __stop: SupportsIndex = sys.maxsize) -> int: ...
    # fmt: on


@runtime_checkable
class SupportsShape(Protocol[scalar_co]):
    r"""We just test for shape, since e.g. tf.Tensor does not have ndim."""

    @property
    def shape(self) -> Sequence[int]:
        """Yield the shape of the array."""

    def __len__(self) -> int:
        """Number of elements along first axis."""

    def __getitem__(self, key: Any) -> Self | scalar_co:
        """Return an element/slice of the table."""


@runtime_checkable
class Array(Protocol[scalar_co]):
    r"""Protocol for array-like objects.

    Compared to a Table (e.g. `pandas.DataFrame` / `pyarrow.Table`), an Array has a single dtype.
    """

    @property
    def ndim(self) -> int:
        r"""Number of dimensions."""

    @property
    def dtype(self) -> scalar_co:
        r"""Yield the data type of the array."""

    @property
    def shape(self) -> Sequence[int]:
        """Yield the shape of the array."""

    def __len__(self) -> int:
        """Number of elements along first axis."""

    def __getitem__(self, key: Any) -> Self:
        """Return an element/slice of the table."""

    # # fmt: off
    # def __len__(self) -> int: ...
    # # @overload
    # # def __getitem__(self: A, key: int) -> Any: ...
    # # @overload
    # # def __getitem__(self: A, key: Sequence[bool]) -> A: ...
    # # @overload
    # # def __getitem__(self: A, key: Sequence[int]) -> A: ...
    # def __getitem__(self: A, key: Any) -> A: ...
    # def __eq__(self: A, other: Any) -> A: ...  # type: ignore[override]
    # def __le__(self: A, other: Any) -> A: ...
    # def __ge__(self: A, other: Any) -> A: ...
    # def __lt__(self: A, other: Any) -> A: ...
    # def __gt__(self: A, other: Any) -> A: ...
    # def __ne__(self: A, other: Any) -> A: ...  # type: ignore[override]
    # def __neg__(self: A) -> A: ...
    # def __invert__(self: A) -> A: ...
    # def __add__(self: A, other: Any) -> A: ...
    # def __radd__(self: A, other: Any) -> A: ...
    # def __iadd__(self: A, other: Any) -> A: ...
    # def __sub__(self: A, other: Any) -> A: ...
    # def __rsub__(self: A, other: Any) -> A: ...
    # def __isub__(self: A, other: Any) -> A: ...
    # def __mul__(self: A, other: Any) -> A: ...
    # def __rmul__(self: A, other: Any) -> A: ...
    # def __imul__(self: A, other: Any) -> A: ...
    # def __truediv__(self: A, other: Any) -> A: ...
    # def __rtruediv__(self: A, other: Any) -> A: ...
    # def __itruediv__(self: A, other: Any) -> A: ...
    # # fmt: on


@runtime_checkable
class Lookup(Protocol[key_contra, value_co]):
    """Mapping/Sequence like generic that is contravariant in Keys."""

    def __contains__(self, key: key_var) -> bool:
        # Here, any Hashable input is accepted.
        """Return True if the map contains the given key."""

    def __getitem__(self, key: key_contra) -> value_co:
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

    def copy(self) -> Self:
        """Return a clone of the hash object."""
