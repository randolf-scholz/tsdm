r"""Generic types for type hints etc."""

__all__ = [
    # Type Variables
    "ObjectType",
    "ReturnType",
    "nnModuleType",
    # Types
    "LookupTable",
    "PathType",
    "NestedType",
    "NullableNestedType",
    "KeyType",
    "ValueType",
]

import logging
import os
from collections.abc import Collection, Mapping
from typing import Any, TypeVar, Union

from torch import nn

__logger__ = logging.getLogger(__name__)

KeyType = TypeVar("KeyType")
r"""TypeVar for ``Mapping`` keys."""

ValueType = TypeVar("ValueType")
r"""TypeVar for ``Mapping`` values."""

PathType = Union[str, os.PathLike[str]]
r"""Type for path-like objects."""

TYPE = TypeVar("TYPE")
r"""Generic type hint"""

ObjectType = TypeVar("ObjectType", bound=object)
r"""Generic type hint for instances."""

ReturnType = TypeVar("ReturnType")
r"""Generic type hint for return values."""

ObjectTypeA = TypeVar("ObjectTypeA")
r"""Generic type hint for instances."""

ObjectTypeB = TypeVar("ObjectTypeB")
r"""Generic type hint for instances."""

LookupTable = dict[str, ObjectType]
r"""Table of objects."""

NestedTypeVar = TypeVar("NestedTypeVar")
r"""Type variable for nested type variables."""

NestedType = Union[
    NestedTypeVar,
    Collection[NestedTypeVar],
    Mapping[Any, Union[NestedTypeVar, Collection[NestedTypeVar]]],
]
r"""Type for nested type variables."""

NullableNestedType = Union[
    None,
    NestedTypeVar,
    Collection[NestedTypeVar],
    Mapping[Any, Union[None, NestedTypeVar, Collection[NestedTypeVar]]],
]
r"""Type for nullable nested type variables."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for nn.Modules."""


# R = TypeVar("R")
#
#
# class AbstractAttribute:
#     __is_abstract_attribute__ = True


# class DummyAttribute:
#     pass
#
#
# class AbstractAttribute(obj: Callable[[Any], R] = None) -> R:
#
#     def __init__(self, obj: Callable[[Any], R] = None):
#         self.obj = obj
#
#     _obj = cast(Any, obj)
#
#     if obj is None:
#         _obj = DummyAttribute()
#     _obj.__is_abstract_attribute__ = True
#     return cast(R, _obj)

# TODO: replace with variadic generics PEP646


#
# class ClassLookupTable(dict[str, CLS]):
#     r"""dict of classes"""


# ModularTable = dict[str, type[T]]
# FunctionalTable = dict[str, Callable[..., S]]
# LookupTable = Union[ModularTable, FunctionalTable, dict[str, Union[type[T], Callable[..., S]]]]
#
# LookupTable = dict[str, TYPE]
# """Generic """

#
# class _Final:
#     """Mixin to prohibit subclassing"""
#
#     __slots__ = ('__weakref__',)
#
#     def __init_subclass__(self, /, *args, **kwds):
#         if '_root' not in kwds:
#             raise TypeError("Cannot subclass special typing classes")
#
# class _Immutable:
#     """Mixin to indicate that object should not be copied."""
#     __slots__ = ()
#
#     def __copy__(self):
#         return self
#
#     def __deepcopy__(self, memo):
#         return self
#
#
# # Internal indicator of special typing constructs.
# # See __doc__ instance attribute for specific docs.
# class _SpecialForm(_Final, _root=True):
#     __slots__ = ('_name', '__doc__', '_getitem')
#
#     def __init__(self, getitem):
#         self._getitem = getitem
#         self._name = getitem.__name__
#         self.__doc__ = getitem.__doc__
#
#     def __getattr__(self, item):
#         if item in {'__name__', '__qualname__'}:
#             return self._name
#
#         raise AttributeError(item)
#
#     def __mro_entries__(self, bases):
#         raise TypeError(f"Cannot subclass {self!r}")
#
#     def __repr__(self):
#         return 'typing.' + self._name
#
#     def __reduce__(self):
#         return self._name
#
#     def __call__(self, *args, **kwds):
#         raise TypeError(f"Cannot instantiate {self!r}")
#
#     def __or__(self, other):
#         return Union[self, other]
#
#     def __ror__(self, other):
#         return Union[other, self]
#
#     def __instancecheck__(self, obj):
#         raise TypeError(f"{self} cannot be used with isinstance()")
#
#     def __subclasscheck__(self, cls):
#         raise TypeError(f"{self} cannot be used with issubclass()")
#
#     @_tp_cache
#     def __getitem__(self, parameters):
#         return self._getitem(self, parameters)

# # @_SpecialForm
# def ClassLookupTable(arg):
#     """Optional type.
#     Optional[X] is equivalent to Union[X, None].
#     """
#     # arg = _type_check(parameters, f"{self} requires a single type.")
#     return dict[str, arg]
#
#
# T = TypeVar("T")
#
#
# class Registry(Generic[T]):
#     def __init__(self) -> None:
#         self._store: dict[str, T] = {}
#
#     def set_item(self, k: str, v: T) -> None:
#         self._store[k] = v
#
#     def get_item(self, k: str) -> T:
#         return self._store[k]


# ModularType = type[TYPE]
# r"""Generic type hint for modular objects."""
#
# FunctionalType = Callable
# r"""Generic type hint for functional objects."""
#
#
# class ModularLookUpTable(Generic[TYPE]):
#     def __getitem__(self, key: TYPE) -> dict[str, type[TYPE]]:
#         ...
#
#
# class FunctionalLookUpTable(Generic[TYPE]):
#     def __getitem__(self, key: TYPE) -> dict[str, type[TYPE]]:
#         ...

# class LookUpTable(Generic[TYPE]):
#     def __getitem__(self, key: TYPE) -> dict[str, TYPE]:
#         ...
#

# ModularLookup =
# """ """

# LookUpType = Final[dict[str, Union[ModularType, FunctionalType]]]
# """Type hint for class/function lookup tables."""
