r"""Utilities for wrapping objects and types, in particular for redirecting methods."""

__all__ = [
    # Protocols
    "IterKeys",
    "IterValues",
    "IterItems",
    # Functions
    "iter_items",
    "iter_keys",
    "iter_values",
]


from collections.abc import Iterator, Mapping
from functools import wraps
from typing import Protocol, overload, override

from tsdm.types.protocols import Map


class IterKeys[K, V](Map[K, V], Protocol):  # K, +V
    r"""Protocol for objects with __iter__ and items()."""

    @override
    def __iter__(self) -> Iterator[K]: ...


class IterValues[K, V](Map[K, V], Protocol):  # K, +V
    r"""Protocol for objects with __iter__ and items()."""

    @override
    def __iter__(self) -> Iterator[V]: ...  # type: ignore[override]


class IterItems[K, V](Map[K, V], Protocol):  # K, +V
    r"""Protocol for objects with __iter__ and items()."""

    @override
    def __iter__(self) -> Iterator[tuple[K, V]]: ...  # type: ignore[override]


@overload
def iter_keys[K, V](obj: type[Map[K, V]], /) -> type[IterKeys[K, V]]: ...
@overload
def iter_keys[K, V](obj: Map[K, V], /) -> IterKeys[K, V]: ...
@overload
def iter_keys[T](obj: T, /) -> T: ...
def iter_keys(obj, /):
    r"""Redirects __iter__ to keys()."""
    base_class: type[Mapping] = obj if isinstance(obj, type) else type(obj)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type, misc]
        r"""A simple Wrapper."""

        def __iter__(self):
            return iter(self.keys())

        def __repr__(self) -> str:
            r"""Representation of the new object."""
            return r"IterKeys@" + super().__repr__()

    assert isinstance(WrappedClass, type)
    assert issubclass(WrappedClass, base_class)

    if isinstance(obj, type):
        return WrappedClass

    try:  # instantiate object
        new_obj = WrappedClass(obj)  # pyright: ignore[reportCallIssue]
    except Exception as exc:
        raise TypeError(f"Could not wrap {obj} with {WrappedClass}") from exc
    return new_obj


@overload
def iter_values[K, V](obj: type[Map[K, V]], /) -> type[IterValues[K, V]]: ...
@overload
def iter_values[K, V](obj: Map[K, V], /) -> IterValues[K, V]: ...
@overload
def iter_values[T](obj: T, /) -> T: ...
def iter_values(obj, /):
    r"""Redirects __iter__ to values()."""
    base_class: type[Mapping] = obj if isinstance(obj, type) else type(obj)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type, misc]
        r"""A simple Wrapper."""

        def __iter__(self):
            return iter(self.values())

        def __repr__(self) -> str:
            r"""Representation of the new object."""
            return r"IterValues@" + super().__repr__()

    assert isinstance(WrappedClass, type)
    assert issubclass(WrappedClass, base_class)

    if isinstance(obj, type):
        return WrappedClass

    try:  # instantiate object
        new_obj = WrappedClass(obj)  # pyright: ignore[reportCallIssue]
    except Exception as exc:
        raise TypeError(f"Could not wrap {obj} with {WrappedClass}") from exc
    return new_obj


@overload
def iter_items[K, V](obj: type[Map[K, V]], /) -> type[IterItems[K, V]]: ...
@overload
def iter_items[K, V](obj: Map[K, V], /) -> IterItems[K, V]: ...
@overload
def iter_items[T](obj: T, /) -> T: ...
def iter_items(obj, /):
    r"""Redirects __iter__ to items()."""
    base_class: type[Mapping] = obj if isinstance(obj, type) else type(obj)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type, misc]
        r"""A simple Wrapper."""

        def __iter__(self):
            return iter(self.items())

        def __repr__(self) -> str:
            r"""Representation of the dataset."""
            return r"IterItems@" + super().__repr__()

    assert isinstance(WrappedClass, type)
    assert issubclass(WrappedClass, base_class)

    if isinstance(obj, type):
        return WrappedClass

    try:  # instantiate object
        new_obj = WrappedClass(obj)  # pyright: ignore[reportCallIssue]
    except Exception as exc:
        raise TypeError(f"Could not wrap {obj} with {WrappedClass}") from exc
    return new_obj
