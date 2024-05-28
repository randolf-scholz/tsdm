r"""Function decorators for wrapping classes with additional functionality."""

__all__ = [
    # ABCs & Protocols
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

from torch import jit, nn
from typing_extensions import Any, TypeVar

from tsdm.config import CONFIG
from tsdm.types.protocols import Dataclass, NTuple, SupportsArray
from tsdm.types.variables import T, torch_module_var
from tsdm.utils.decorators.base import decorator
from tsdm.utils.pprint import (
    repr_array,
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_sequence,
    repr_set,
    repr_shortform,
)

Seq = TypeVar("Seq", bound=Sequence)
Map = TypeVar("Map", bound=Mapping)
Dtc = TypeVar("Dtc", bound=Dataclass)
Ntp = TypeVar("Ntp", bound=NTuple)
Set = TypeVar("Set", bound=AbstractSet)


@decorator
def pprint_sequence(cls: type[Seq], /, **kwds: Any) -> type[Seq]:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Sequence):
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_sequence, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_mapping(cls: type[Map], /, **kwds: Any) -> type[Map]:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Mapping):
        raise TypeError(f"Expected Mapping type, got {cls}.")
    cls.__repr__ = partialmethod(repr_mapping, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_dataclass(cls: type[Dtc], /, **kwds: Any) -> type[Dtc]:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Dataclass):  # type: ignore[misc]
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_dataclass, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_namedtuple(cls: type[Ntp], /, **kwds: Any) -> type[Ntp]:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, NTuple):  # type: ignore[misc]
        raise TypeError(f"Expected NamedTuple type, got {cls}.")
    cls.__repr__ = partialmethod(repr_namedtuple, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_set(cls: type[Set], /, **kwds: Any) -> type[Set]:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, AbstractSet):
        raise TypeError(f"Expected Set type, got {cls}.")
    cls.__repr__ = partialmethod(repr_set, **kwds)  # type: ignore[assignment]
    return cls


@decorator
def pprint_repr(cls: type[T], /, **kwds: Any) -> type[T]:
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
    elif issubclass(cls, type):
        repr_func = repr_shortform
    else:
        raise TypeError(f"Unsupported type {cls}.")

    cls.__repr__ = partialmethod(repr_func, **kwds)  # type: ignore[assignment]
    return cls  # pyright: ignore[reportReturnType]


def autojit(base_class: type[torch_module_var], /) -> type[torch_module_var]:
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
    assert isinstance(base_class, type)
    assert issubclass(base_class, nn.Module)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type,misc]  # pylint: disable=too-few-public-methods
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> torch_module_var:  # type: ignore[misc]
            # Note: If __new__() does not return an instance of cls,
            # then the new instance's __init__() method will not be invoked.
            instance: torch_module_var = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted: torch_module_var = jit.script(instance)
                return scripted
            return instance

    assert isinstance(WrappedClass, type)
    assert issubclass(WrappedClass, base_class)
    return WrappedClass


def implements(*protocols: type) -> Callable[[type[T]], type[T]]:
    r"""Check if class implements a set of protocols."""

    def __wrapper(cls: type[T], /) -> type[T]:
        for protocol in protocols:
            if not issubclass(cls, protocol):
                raise TypeError(f"{cls} does not implement {protocol}")
        return cls

    return __wrapper
