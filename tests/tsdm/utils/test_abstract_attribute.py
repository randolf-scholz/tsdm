# #!/usr/bin/env python
"""Test the abstract attribute decorator."""
#
# from abc import ABCMeta
# from typing import Any, Callable, Generic, TypeAlias, TypeVar, cast
#
# R = TypeVar("R")
#
# import typing
# from types import GenericAlias
#
# # _GenericAlias = typing._GenericAlias
# # _SpecialForm = typing._SpecialForm
# # _type_check = typing._type_check
#
#
# # @_SpecialForm
# # def Abstract(self, parameters):
# #     """A special typing construct to mark an attribute as abstract."""
# #     item = _type_check(parameters, f"{self._name} accepts only a single type.")
# #     return _GenericAlias(self, (item,))
# # #
#
#
# class Abstract(Generic[R]):
#     def __class_getitem__(cls, klass: R) -> R:
#         return klass
#
#
# foo: Abstract[int] = 2
# bar: int = foo
# reveal_type(foo)
# reveal_type(bar)
#
#
# class DummyAttribute:
#     pass
#
#
# def abstract_attribute(obj: Callable[[Any], R] = NotImplemented) -> R:
#     _obj = cast(Any, obj)
#     if obj is NotImplemented:
#         _obj = DummyAttribute()
#     _obj.__is_abstract_attribute__ = True
#     return cast(R, _obj)
#
#
# class AbstractFooTyped(metaclass=ABCMeta):
#     @abstract_attribute
#     def bar(self) -> int:
#         ...
#
#
# class FooTyped(AbstractFooTyped):
#     def __init__(self):
#         # skipping assignment (which is required!) to demonstrate
#         # that it works independent of when the assignment is made
#         pass
#
#
# f_typed = FooTyped()
# _ = f_typed.bar + "test"  # Mypy: Unsupported operand types for + ("int" and "str")
#
#
# FooTyped.bar = "test"  # Mypy: Incompatible types in assignment (expression has type "str", variable has type "int")
# FooTyped.bar + "test"  # Mypy: Unsupported operand types for + ("int" and "str")
#

# from abc import ABC, abstractmethod
#
#
# class Foo(ABC):
#     @property
#     @abstractmethod
#     def myattr(self) -> int:
#         ...
#
#
# class Bar(Foo):
#     myattr: int = 0
#
#
# class Baz(Foo):
#     @property
#     def myattr(self) -> int:
#         return 0


# if __name__ == "__main__":
#     pass
