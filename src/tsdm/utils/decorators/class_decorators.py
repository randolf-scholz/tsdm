r"""Function decorators for wrapping classes with additional functionality."""

__all__ = [
    # Functions
    "autojit",
    "implements",
    "pprint_dataclass",
    "pprint_mapping",
    "pprint_namedtuple",
    "pprint_repr",
    "pprint_sequence",
    "pprint_set",
]

from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from functools import partialmethod, wraps
from typing import Any, Self

from torch import jit, nn

from tsdm.config import CONFIG
from tsdm.types.protocols import Dataclass, NTuple, SupportsArray
from tsdm.utils.decorators.base import PolymorphicClassDecorator, decorator
from tsdm.utils.pprint import (
    repr_array,
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_sequence,
    repr_set,
    repr_shortform,
)

# region workaround mypy bug -----------------------------------------------------------
# FIXME: https://github.com/python/mypy/issues/17191
pprint_sequence: PolymorphicClassDecorator[Any]  # pyright: ignore
pprint_mapping: PolymorphicClassDecorator[Any]  # pyright: ignore
pprint_set: PolymorphicClassDecorator[Any]  # pyright: ignore
pprint_dataclass: PolymorphicClassDecorator[Any]  # pyright: ignore
pprint_namedtuple: PolymorphicClassDecorator[Any]  # pyright: ignore
pprint_repr: PolymorphicClassDecorator[Any]  # pyright: ignore
# endregion workaround mypy bug --------------------------------------------------------


@decorator
def pprint_sequence[Seq: Sequence](cls: type[Seq], /, **kwds: Any) -> type[Seq]:
    # def pprint_sequence[Seq: type[Sequence]](cls: Seq, /, **kwds: Any) -> Seq:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Sequence):
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_sequence, **kwds)  # type: ignore[assignment]
    return cls


@decorator
# def pprint_mapping[Map: Mapping](cls: type[Map], /, **kwds: Any) -> type[Map]:
def pprint_mapping[Map: type[Mapping]](cls: Map, /, **kwds: Any) -> Map:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Mapping):
        raise TypeError(f"Expected Mapping type, got {cls}.")
    cls.__repr__ = partialmethod(repr_mapping, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_set[Set: AbstractSet](cls: type[Set], /, **kwds: Any) -> type[Set]:
    # def pprint_set[Set: type[AbstractSet]](cls: Set, /, **kwds: Any) -> Set:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, AbstractSet):
        raise TypeError(f"Expected Set type, got {cls}.")
    cls.__repr__ = partialmethod(repr_set, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_dataclass[Dtc: Dataclass](cls: type[Dtc], /, **kwds: Any) -> type[Dtc]:
    # def pprint_dataclass[Dtc: type[Dataclass]](cls: Dtc, /, **kwds: Any) -> Dtc:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Dataclass):  # type: ignore[misc]
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_dataclass, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_namedtuple[Ntp: NTuple](cls: type[Ntp], /, **kwds: Any) -> type[Ntp]:
    # def pprint_namedtuple[Ntp: type[NTuple]](cls: Ntp, /, **kwds: Any) -> Ntp:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, NTuple):  # type: ignore[misc]
        raise TypeError(f"Expected NamedTuple type, got {cls}.")
    cls.__repr__ = partialmethod(repr_namedtuple, **kwds)  # type: ignore[assignment]
    return cls


@decorator
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


def autojit[M: nn.Module](base_class: type[M], /) -> type[M]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule: ...


        model = jit.script(MyModule())

    and

    .. code-block:: python

        @autojit
        class MyModule: ...


        model = MyModule()

    are (roughly?) equivalent
    """
    if not isinstance(base_class, type):
        raise TypeError("Expected a class.")
    if not issubclass(base_class, nn.Module):
        raise TypeError("Expected a subclass of nn.Module.")

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type,misc]
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            # Note: If __new__() does not return an instance of cls,
            #   then the new instance's __init__() method will not be invoked.
            instance = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted = jit.script(instance)
                return scripted  # type: ignore[return-value]
            return instance  # type: ignore[return-value]

    if not isinstance(WrappedClass, type):
        raise TypeError(f"Expected a class, got {WrappedClass}.")
    if not issubclass(WrappedClass, base_class):
        raise TypeError(f"Expected {WrappedClass} to be a subclass of {base_class}.")

    return WrappedClass


def implements[T](*protocols: type) -> Callable[[type[T]], type[T]]:
    r"""Check if class implements a set of protocols."""

    def __wrapper(cls: type[T], /) -> type[T]:
        for protocol in protocols:
            if not issubclass(cls, protocol):
                raise TypeError(f"{cls} does not implement {protocol}")
        return cls

    return __wrapper
