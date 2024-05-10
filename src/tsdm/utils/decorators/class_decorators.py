r"""Function decorators for wrapping classes with additional functionality."""

__all__ = [
    # ABCs & Protocols
    "ClassDecorator",
    "ClassDecoratorFactory",
    # Functions
    "autojit",
    "implements",
    "pprint_sequence",
    "pprint_mapping",
    "pprint_dataclass",
    "pprint_namedtuple",
    "pprint_repr",
]
from collections.abc import Callable, Mapping, Sequence
from functools import partialmethod, wraps

from torch import jit, nn
from typing_extensions import Any, Protocol, TypeVar

from tsdm.config import CONFIG
from tsdm.types.protocols import Dataclass, NTuple, SupportsArray
from tsdm.types.variables import Cls, torch_module_var
from tsdm.utils.decorators._decorators import decorator
from tsdm.utils.pprint import (
    repr_array,
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_sequence,
    repr_shortform,
)


class ClassDecorator(Protocol[Cls]):
    r"""Class Decorator Protocol that preserves type."""

    def __call__(self, cls: Cls, /) -> Cls:
        r"""Decorate a class."""
        ...


class ClassDecoratorFactory(Protocol[Cls]):
    r"""Class Decorator Factory Protocol that preserves type."""

    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[Cls], Cls]:
        r"""Create a class decorator."""
        ...


SeqType = TypeVar("SeqType", bound=type[Sequence])
MapType = TypeVar("MapType", bound=type[Mapping])
DtcType = TypeVar("DtcType", bound=type[Dataclass])
NtpType = TypeVar("NtpType", bound=type[NTuple])


@decorator
def pprint_sequence(cls: SeqType, /, **kwds: Any) -> SeqType:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Sequence):
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_sequence, **kwds)  # pyright: ignore[reportAttributeAccessIssue]
    return cls


@decorator
def pprint_mapping(cls: MapType, /, **kwds: Any) -> MapType:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Mapping):
        raise TypeError(f"Expected Mapping type, got {cls}.")
    cls.__repr__ = partialmethod(repr_mapping, **kwds)  # pyright: ignore[reportAttributeAccessIssue]
    return cls


@decorator
def pprint_dataclass(cls: DtcType, /, **kwds: Any) -> DtcType:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Dataclass):
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_dataclass, **kwds)  # pyright: ignore[reportAttributeAccessIssue]
    return cls


@decorator
def pprint_namedtuple(cls: NtpType, /, **kwds: Any) -> NtpType:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, NTuple):
        raise TypeError(f"Expected NamedTuple type, got {cls}.")
    cls.__repr__ = partialmethod(repr_namedtuple, **kwds)
    return cls


@decorator
def pprint_repr(cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not isinstance(cls, type):
        raise TypeError("Must be a class!")

    repr_func: Callable[..., str]

    if issubclass(cls, Dataclass):
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

    cls.__repr__ = partialmethod(repr_func, **kwds)  # pyright: ignore[reportAttributeAccessIssue]
    return cls


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


def implements(*protocols: type) -> Callable[[Cls], Cls]:
    r"""Check if class implements a set of protocols."""

    def __wrapper(cls: Cls, /) -> Cls:
        for protocol in protocols:
            if not issubclass(cls, protocol):
                raise TypeError(f"{cls} does not implement {protocol}")
        return cls

    return __wrapper
