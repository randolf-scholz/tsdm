#!/usr/bin/env python
r"""Test type covariance."""

from collections.abc import Hashable
from typing import TypeVar

key = TypeVar("key", bound=Hashable)
key_co = TypeVar("key_co", bound=Hashable, covariant=True)
key_contra = TypeVar("key_contra", bound=Hashable, contravariant=True)

value = TypeVar("value")
value_co = TypeVar("value_co", covariant=True)
value_contra = TypeVar("value_contra", contravariant=True)


# from typing import Any, Mapping, Union, TypeVar
#
# # S = TypeVar("S", bound=Union[str, int])
# S = TypeVar("S", bound=Hashable)
#
#
# def f(arg: Mapping[str, Any]) -> None:
#     pass
#
#
# def g(arg: Mapping[S, Any]) -> S:
#     return next(iter(arg))
#
#
# f({"b": "c"})
# g({"b": "c"})
# d = {"b": "c"}
# e = {(1,): 2}
# f(d)
# g(e)
# n = g(d)  # <--- Line 8
# reveal_type(n)


# from typing import Any, Mapping, Union
# def f(arg: Mapping[      str      , Any]) -> None: pass
# def g(arg: Mapping[Union[str, int], Any]) -> None: pass
# f({'b': 'c'})
# g({'b': 'c'})
# d = {'b': 'c'}
# f(d)
# g(d)  # <--- Line 8


# def bar(x: str | int) -> None: pass
# def f(arg: Callable[[str], Any]) -> None: pass


# def foo(x: str) -> None: pass
# def g(arg: Callable[[str | int], Any]) -> None: pass
# def h(arg: Callable[[str], Any]) -> None: return
#
#
# f(bar)
# g(foo)

# def foo(x: str|int) -> None: pass
# def bar(x: Callable[[str], None]) -> None: pass
# def baz(x: Callable[[str|int], None]) -> None: pass
# def hoo(x: Callable[[str|int], None]) -> None: return bar(x)
#
# x: str|int = "1"
# y: Callable[[str], None] = foo
# z: Callable[[Callable[[str | int], None]], None] = bar
#
# d: dict[str, None] = {"1" : None}
#
#
# d_get: Callable[[str], None] = d.__getitem__ # bar
#
#
#
#
#
#
# def g(arg: Mapping[Union[str, int], Any]) -> None: pass

#
# # contravariance of Callable
# def int_fun(x: int) -> None: pass
# def str_or_int_fun(x: str | int) -> None: pass
#
# str_fun_obj: Callable[[str], None] = str_fun
# int_fun_obj: Callable[[int], None] = int_fun
# str_or_int_fun_obj: Callable[[str | int], None] = str_or_int_fun
# str_or_int_fun_as_str_fun: Callable[[str], None] = str_or_int_fun
# str_or_int_fun_as_int_fun: Callable[[int], None] = str_or_int_fun
#
# # covariance of nested callable.
# def foo(x: Callable[[str], None]) -> None: pass
# def bar(x: Callable[[str | int], None]) -> None: pass
#
# foo_obj: Callable[[Callable[[str], None]], None] = foo  # ✔
# bar_obj: Callable[[Callable[[str | int], None]], None] = bar  # ✔
# foo_as_bar: Callable[[Callable[[str | int], None]], None] = foo  # ✔ (chain rule!)
#
#
# from typing import Any, Mapping, Union
# def f(arg: Callable[[str],       Any]) -> None: pass
# def g(arg: Callable[[str | int], Any]) -> None: pass
#
# h: Callable[[Callable[[str], str]], None] = g
#
# d = {'b': 'c'}
# item_access: Callable[[str|int], Any] = d.__getitem__
#
# f(d.__getitem__)
# g(d.__getitem__)  # <--- Line 8
# h(augmented)
#

#
# class Foo:
#
#     def internal(self, x: str | int) -> None: pass
#
#     def __call__(self, x: Callable[[str | int], None]) -> None:
#         pass
#
#
# class Bar(Foo):
#     def __call__(self, x: Callable[[str], None]) -> None:
#         return super().__call__(x)


# d: dict[str, str] = {"a" : "b"}
# e: dict[str|int, str] = {"a" : "b", 1 : "0"}
# g(d.__getitem__)  # <--- Line 8
# f(e.__getitem__)  # <--- Line 9
# reveal_type(e.__getitem__)

# f({'b': 'c'}.__getitem__)
# g({'b': 'c'}.__getitem__)
#
# f(d.__getitem__)


#
# def test_mapping_covariant() -> None:
#     r"""Test that Mapping is covariant."""
#
#     class InInMapping(Mapping[key, value]):
#         r"""Mapping with input key and value types."""
#
#     class InCoMapping(Mapping[key, value_co]):
#         r"""Mapping with input key type and covariant value type."""
#
#
#     # def unpack(x: Mapping[Hashable, value_co]) -> list[value_co]:
#     #     return list(x.values())
#     #
#     #
#     # d: dict[str, int] = {"a": 1}
#     # values: list[int] = unpack(d)  # ✘ expected Mapping[Hashable, int]
#     #
#
#
#
#
#     _Value = TypeVar("_Value")
#
#     def lower_clean_dict_keys(dict: Mapping[str, _Value]) -> Mapping[str, _Value]:
#         return {k.lower().strip(): v for k, v in dict.items()}
#
#
#     def foo(s: Hashable) -> None:
#         pass
#
#     x: Callable[[str], None] = foo
#
#     # chain rule
#     def bar(f: Callable[[str], None]) -> None:
#         pass
#
#     y: Callable[[Callable[[Hashable], None]], None] = bar
#
#
#     z: Hashable = "a"  ✔
#
# from typing import TypeVar, Mapping, Hashable
#
# value_co = TypeVar("value_co", covariant=True)
#
# HashableT = TypeVar("HashableT")
#
# def unpack(x: Mapping[Hashable, value_co]) -> list[value_co]:
#     return list(x.values())


# value_co = TypeVar("value_co", covariant=True)
# HashableT = TypeVar("HashableT")
# def unpack(x: Mapping[HashableT, value_co]) -> list[value_co]:
#     return list(x.values())
# d: dict[str, int] = {"a": 1}
# values: list[int] = unpack(d)  # ✘ expected Mapping[Hashable, int]


# from typing import Any, Mapping, Union, TypeVar
#
# StrVar = TypeVar("StrVar", bound=str)
# StrOrIntVar = TypeVar("StrOrIntVar", bound=Union[str, int])

# def f(arg: Mapping[      StrVar      , Any]) -> None: pass
# def g(arg: Mapping[StrOrIntVar, Any]) -> None: pass
# f({'b': 'c'})
# g({'b': 'c'})
# d = {'b': 'c'}
# f(d)
# g(d)


# class CoInMapping(Mapping[key_co, value]):
#     r"""Mapping with covariant key type and input value type."""

# class ContraInMapping(Mapping[key_contra, value]):
#     r"""Mapping with contravariant key type and input value type."""

# class InContraMapping(Mapping[key, value_contra]):
#     r"""Mapping with input key type and contravariant value type."""

# class CoCoMapping(Mapping[key_co, value_co]):
#     r"""Covariant mapping."""

# class ContraContraMapping(Mapping[key_contra, value_contra]):
#     r"""Contravariant mapping."""

# class CoContraMapping(Mapping[key_co, value_contra]):
#     r"""Co-contravariant mapping."""

# class ContraCoMapping(Mapping[key_contra, value_co]):
#     r"""Contravariant-co mapping."""


#
# if __name__ == "__main__":
#     _main()
