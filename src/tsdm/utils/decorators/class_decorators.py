r"""Function decorators for wrapping classes with additional functionality."""

__all__ = [
    # Functions
    "implements",
    "pprint_dataclass",
    "pprint_mapping",
    "pprint_namedtuple",
    "pprint_repr",
    "pprint_sequence",
    "pprint_set",
]

from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from functools import partialmethod
from typing import Any

from tsdm.pprint import (
    repr_array,
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_sequence,
    repr_set,
    repr_shortform,
)
from tsdm.types.arrays import SupportsArray
from tsdm.types.protocols import Dataclass, NTuple
from tsdm.utils.decorators.base import PolymorphicClassDecorator, decorator

# region workaround mypy bug -----------------------------------------------------------
# FIXME: https://github.com/python/mypy/issues/17191
pprint_sequence: PolymorphicClassDecorator[Any]  # pyright: ignore[reportRedeclaration]
pprint_mapping: PolymorphicClassDecorator[Any]  # pyright: ignore[reportRedeclaration]
pprint_set: PolymorphicClassDecorator[Any]  # pyright: ignore[reportRedeclaration]
pprint_dataclass: PolymorphicClassDecorator[Any]  # pyright: ignore[reportRedeclaration]
pprint_namedtuple: PolymorphicClassDecorator[Any]  # pyright: ignore[reportRedeclaration]
pprint_repr: PolymorphicClassDecorator[Any]  # pyright: ignore[reportRedeclaration]
# endregion workaround mypy bug --------------------------------------------------------


@decorator  # type: ignore[no-redef]
def pprint_sequence[Seq: Sequence](cls: type[Seq], /, **kwds: Any) -> type[Seq]:
    # def pprint_sequence[Seq: type[Sequence]](cls: Seq, /, **kwds: Any) -> Seq:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Sequence):
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_sequence, **kwds)  # type: ignore[assignment]
    return cls


@decorator  # type: ignore[no-redef]
def pprint_mapping[Map: Mapping](cls: type[Map], /, **kwds: Any) -> type[Map]:
    # def pprint_mapping[Map: type[Mapping]](cls: Map, /, **kwds: Any) -> Map:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Mapping):
        raise TypeError(f"Expected Mapping type, got {cls}.")
    cls.__repr__ = partialmethod(repr_mapping, **kwds)  # type: ignore[assignment]
    return cls


@decorator  # type: ignore[no-redef]
def pprint_set[Set: AbstractSet](cls: type[Set], /, **kwds: Any) -> type[Set]:
    # def pprint_set[Set: type[AbstractSet]](cls: Set, /, **kwds: Any) -> Set:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, AbstractSet):
        raise TypeError(f"Expected Set type, got {cls}.")
    cls.__repr__ = partialmethod(repr_set, **kwds)  # type: ignore[assignment]
    return cls


# NOTE: Less specific than needed due to
#   https://github.com/microsoft/pyright/issues/8681#issuecomment-2271979444
@decorator  # type: ignore[no-redef]
def pprint_dataclass[T](cls: type[T], /, **kwds: Any) -> type[T]:
    # def pprint_dataclass[Dtc: Dataclass](cls: type[Dtc], /, **kwds: Any) -> type[Dtc]: ...
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Dataclass):  # type: ignore[misc]
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_dataclass, **kwds)  # type: ignore[assignment]
    return cls  # type: ignore[return-value]


@decorator  # type: ignore[no-redef]
def pprint_namedtuple[Ntp: NTuple](cls: type[Ntp], /, **kwds: Any) -> type[Ntp]:
    # def pprint_namedtuple[Ntp: type[NTuple]](cls: Ntp, /, **kwds: Any) -> Ntp:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, NTuple):  # type: ignore[misc]
        raise TypeError(f"Expected NamedTuple type, got {cls}.")
    cls.__repr__ = partialmethod(repr_namedtuple, **kwds)  # type: ignore[assignment]
    return cls


@decorator  # type: ignore[no-redef]
def pprint_repr[T](cls: type[T], /, **kwds: Any) -> type[T]:
    # def pprint_repr[Cls: type](cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not isinstance(cls, type):
        raise TypeError("Must be a class!")

    repr_func: Callable[..., str]

    if issubclass(cls, Dataclass):  # type: ignore[misc]
        repr_func = repr_dataclass
    elif issubclass(cls, NTuple):  # type: ignore[misc]
        repr_func = repr_namedtuple
    elif issubclass(cls, Mapping):
        repr_func = repr_mapping
    elif issubclass(cls, SupportsArray):
        repr_func = repr_array
    elif issubclass(cls, Sequence):
        repr_func = repr_sequence
    elif issubclass(cls, AbstractSet):
        repr_func = repr_set
    elif issubclass(cls, type):
        repr_func = repr_shortform
    else:
        raise TypeError(f"Unsupported type {cls}.")

    cls.__repr__ = partialmethod(repr_func, **kwds)  # type: ignore[assignment]
    return cls


def implements[T](*protocols: type) -> Callable[[type[T]], type[T]]:
    r"""Check if class implements a set of protocols."""

    def __wrapper(cls: type[T], /) -> type[T]:
        for protocol in protocols:
            if not issubclass(cls, protocol):
                raise TypeError(f"{cls} does not implement {protocol}")
        return cls

    return __wrapper
