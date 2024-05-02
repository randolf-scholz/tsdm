r"""Ulities for wrapping objects and types, in particular for redirecting methods."""

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

from typing_extensions import Protocol, overload, override

from tsdm.types.protocols import MappingProtocol
from tsdm.types.variables import K, T, V_co


class IterKeys(MappingProtocol[K, V_co], Protocol[K, V_co]):
    r"""Protocol for objects with __iter__ and items()."""

    @override
    def __iter__(self) -> Iterator[K]: ...


class IterValues(MappingProtocol[K, V_co], Protocol[K, V_co]):
    r"""Protocol for objects with __iter__ and items()."""

    @override
    def __iter__(self) -> Iterator[V_co]: ...  # type: ignore[override]


class IterItems(MappingProtocol[K, V_co], Protocol[K, V_co]):
    r"""Protocol for objects with __iter__ and items()."""

    @override
    def __iter__(self) -> Iterator[tuple[K, V_co]]: ...  # type: ignore[override]


@overload
def iter_keys(obj: type[MappingProtocol[K, V_co]], /) -> type[IterKeys[K, V_co]]: ...
@overload
def iter_keys(obj: MappingProtocol[K, V_co], /) -> IterKeys[K, V_co]: ...
@overload
def iter_keys(obj: T, /) -> T: ...
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
        new_obj = WrappedClass(obj)
    except Exception as exc:
        raise TypeError(f"Could not wrap {obj} with {WrappedClass}") from exc
    return new_obj


@overload
def iter_values(
    obj: type[MappingProtocol[K, V_co]], /
) -> type[IterValues[K, V_co]]: ...
@overload
def iter_values(obj: MappingProtocol[K, V_co], /) -> IterValues[K, V_co]: ...
@overload
def iter_values(obj: T, /) -> T: ...
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
        new_obj = WrappedClass(obj)
    except Exception as exc:
        raise TypeError(f"Could not wrap {obj} with {WrappedClass}") from exc
    return new_obj


@overload
def iter_items(obj: type[MappingProtocol[K, V_co]], /) -> type[IterItems[K, V_co]]: ...
@overload
def iter_items(obj: MappingProtocol[K, V_co], /) -> IterItems[K, V_co]: ...
@overload
def iter_items(obj: T, /) -> T: ...
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
        new_obj = WrappedClass(obj)
    except Exception as exc:
        raise TypeError(f"Could not wrap {obj} with {WrappedClass}") from exc
    return new_obj
