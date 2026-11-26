r"""Small mixin protocols."""

__all__ = [
    # Protocols
    "SupportsBool",
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsLenAndGetItem",
    "SupportsSlicing",
    # Comparison operations
    "SupportsEquality",
    "SupportsComparison",
    # Mixins
    "HasDType",
    "HasDTypes",
    "HasDevice",
    "HasNDim",
    "HasShape",
    "SupportsArray",
    "SupportsArrayUfunc",
    "SupportsDataFrame",
    "SupportsItem",
    "SupportsRound",
    # Protocols
    "Seq",
    # Runtime checks
    "implements",
    "protocol_exclude",
]


from collections.abc import Collection, Iterator
from types import GenericAlias
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    _ProtocolMeta as ProtocolMeta,
    get_origin,
    is_protocol,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray


class SupportsEquality[ResultT](Protocol):
    r"""Protocol for objects that support equality and inequality comparisons."""

    # Note: really no good choice for the argument type here,
    #   as most built-ins will require object argument, but DSLs will restrict to their own type.
    #   so just use Any.
    # equality ==
    def __eq__(self, other: Any, /) -> ResultT: ...  # type: ignore
    # inequality !=
    def __ne__(self, other: Any, /) -> ResultT: ...  # type: ignore


class SupportsComparison[ComparableT, ResultT](Protocol):
    r"""Protocol for objects that support comparison operations."""

    # comparisons (element-wise)
    def __le__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __ge__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __lt__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __gt__(self, other: Self | ComparableT, /) -> ResultT: ...


@runtime_checkable
class SupportsBool(Protocol):
    r"""Protocol for types that support boolean operations."""

    def __bool__(self) -> bool: ...


@runtime_checkable
class SupportsGetItem[K, V](Protocol):  # -K, +V
    r"""Protocol for objects that support `__getitem__`."""

    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsKeysAndGetItem[K, V](Protocol):  # K, +V
    r"""Protocol for objects that support `__getitem__` and `keys`."""

    def keys(self) -> Collection[K]: ...

    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsLenAndGetItem[V](Protocol):  # +V
    r"""Protocol for objects with integer indexing and `__len__`."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> V: ...


class SupportsSlicing[V](SupportsGetItem[int, V], Protocol):  # +V
    r"""Protocol for objects that support slicing with integer indices."""

    @overload
    def __getitem__(self, index: int, /) -> V: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...


@runtime_checkable
class HasShape(Protocol):
    r"""Protocol for objects with a `shape` property containing integer dimensions."""

    @property
    def shape(self) -> tuple[int, ...]: ...


@runtime_checkable
class HasNDim(Protocol):
    r"""Protocol for objects with an integer `ndim` property."""

    @property
    def ndim(self) -> int: ...


@runtime_checkable
class HasDevice[DeviceT = Any](Protocol):
    r"""Protocol for objects that support `device`."""

    @property
    def device(self) -> DeviceT: ...


@runtime_checkable
class HasDType[DTypeT = Any](Protocol):
    r"""Protocol for objects with a `dtype` property.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `polars.Series`
        - `torch.Tensor`

        Counterexamples include `pandas.DataFrame`, `polars.DataFrame`, and
        `pyarrow.Array`, which do not expose a `dtype` property.
    """

    @property
    def dtype(self) -> DTypeT: ...


@runtime_checkable
class HasDTypes[DTypeT = Any](Protocol):
    r"""Protocol for objects with a `dtypes` property.

    Examples:
        - `pandas.DataFrame`
        - `polars.DataFrame`
    """

    @property
    def dtypes(self) -> Collection[DTypeT]: ...


@runtime_checkable
class SupportsArray[ScalarT: np.generic](Protocol):
    r"""Protocol for objects that support `__array__`.

    References:
        - [NumPy array C API](https://numpy.org/doc/stable/reference/c-api/array.html)
    """

    def __array__(self) -> NDArray[ScalarT]:
        r"""Return a NumPy array representation of the object."""
        ...


@runtime_checkable
class SupportsArrayUfunc(SupportsArray, Protocol):
    r"""Protocol for objects that support `__array_ufunc__`.

    NumPy ufuncs such as `numpy.exp` dispatch to this method. Examples include
    `pandas.Series` and `pandas.DataFrame`.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Index`
        - `pandas.Series`
        - `polars.Series`

        Counterexamples include `polars.DataFrame`, `pyarrow.Array`,
        `pyarrow.Table`, and `torch.Tensor`.

    References:
        - [NumPy universal functions](https://numpy.org/doc/stable/reference/ufuncs.html)
    """

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        /,
        *inputs: Any,
        **kwargs: Any,
    ) -> Self:
        r"""Return the array resulting from applying the ufunc."""
        ...


@runtime_checkable
class SupportsDataFrame[DataFrameT = Any](Protocol):
    r"""Protocol for objects that support `__dataframe__`.

    References:
        - [DataFrame interchange protocol](https://data-apis.org/dataframe-protocol/latest/index.html)
    """

    def __dataframe__(self) -> DataFrameT: ...


@runtime_checkable
class SupportsItem[Scalar](Protocol):  # +T
    r"""Protocol for objects that support `item()`.

    For tensors, `item()` returns the scalar value when there is exactly one
    element and raises an error otherwise, including for empty tensors.
    """

    def item(self) -> Scalar: ...


@runtime_checkable
class SupportsRound(Protocol):
    r"""Protocol for objects with a `round()` method accepting optional `decimals`."""

    # FIXME: https://github.com/python/typing/discussions/1782
    @overload
    def round(self) -> Self: ...
    @overload
    def round(self, *, decimals: int) -> Self: ...


class protocol_exclude(ProtocolMeta):
    r"""Exclude types from runtime protocol checks via the `excluded` class keyword.

    Subclasses inherit the exclusions of their bases.
    """

    _abc_excluded_types: tuple[type, ...]

    # Mypy disallows Self in metaclasses.
    def __new__[MetaT: protocol_exclude](  # ruff: ignore[PYI019]
        cls: type[MetaT],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        excluded: tuple[type, ...] = (),
        **kwargs: Any,
    ) -> MetaT:
        inherited = (
            excluded_type
            for base in bases
            if isinstance(base, protocol_exclude)
            for excluded_type in base._abc_excluded_types  # ruff: ignore[SLF001]
        )
        # typing skips names prefixed with _abc_ when it collects protocol members.
        namespace["_abc_excluded_types"] = (*inherited, *excluded)
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __instancecheck__(self, instance: object, /) -> bool:
        if isinstance(instance, self._abc_excluded_types):
            return False
        return super().__instancecheck__(instance)

    def __subclasscheck__(self, subclass: type, /) -> bool:
        if issubclass(subclass, self._abc_excluded_types):
            return False
        return super().__subclasscheck__(subclass)


@runtime_checkable
class Seq[T](Protocol, metaclass=protocol_exclude, excluded=(str, bytes)):  # +T
    r"""Lightweight protocol version of `collections.abc.Sequence`.

    The membership signature accepts any `object`, which excludes `str` and
    `bytes` in structural type checks that enforce the parameter type.
    Runtime `isinstance` and `issubclass` checks also exclude both types.

    References:
        - [Typeshed typing stubs](https://github.com/python/typeshed/blob/main/stdlib/typing.pyi)
        - [CPython collection ABCs](https://github.com/python/cpython/blob/main/Lib/_collections_abc.py)
    """

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __contains__(self, value: object, /) -> bool: ...
    @overload
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    def __getitem__(self, index: slice, /) -> Seq[T]: ...


def implements[T](cls: type[T], /, *protos: type | GenericAlias) -> type[T]:
    r"""Check a class against runtime-checkable protocols and return the class.

    Parameterized generic aliases are checked using their origin. Protocols
    without runtime checking enabled are accepted without a subclass check.

    Args:
        cls: Class to check.
        *protos: Protocol types or parameterized generic aliases to check against.

    Returns:
        The input class, unchanged.

    Raises:
        TypeError: If an argument in `protos` is not a protocol, or if a protocol
            does not support `issubclass` checks.
        AssertionError: If `cls` fails a runtime subclass check. These assertions
            are skipped when Python is run with optimization enabled.
    """
    for proto in protos:
        if isinstance(proto, GenericAlias):
            proto = get_origin(proto)  # ruff: ignore[PLW2901]

        if not (isinstance(proto, type) and is_protocol(proto)):
            raise TypeError(f"{proto=} is not a protocol type.")

        if bool(getattr(proto, "_is_runtime_protocol", False)):
            assert issubclass(cls, proto)

    return cls
