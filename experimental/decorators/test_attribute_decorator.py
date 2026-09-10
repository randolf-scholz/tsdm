r"""Tests the attribute decorator."""

from collections.abc import Callable as Fn
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ClassVar, Optional, Self, cast, overload

import pytest


class _AttrMeta(type):
    r"""Metaclass for attribute decorators."""

    def __call__[T, R](cls, func: Fn[[T], R], /) -> R:
        r"""Create a decorator that converts method to attribute."""
        attr_ = super().__call__(func)
        wrapper = wraps(func, updated=())
        attr = cast("R", wrapper(attr_))
        return attr


def test_attribute() -> None:
    class Foo:
        @attribute
        def bar(self) -> int:
            return 42

    obj = Foo()
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload is attr.SENTINEL

    # test __get__
    assert obj.bar == 42
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload == 42

    # test __set__
    obj.bar = 24
    assert obj.bar == 24
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload == 24

    # test __delete__
    del obj.bar
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload is attr.DELETED
    with pytest.raises(AttributeError):
        assert obj.bar is not object()

    # test __set__
    obj.bar = 24
    assert obj.bar == 24
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload == 24


@dataclass
class attribute[T, R](metaclass=_AttrMeta):
    r"""Create a decorator that converts method to attribute."""

    SENTINEL: ClassVar[Any] = object()
    DELETED: ClassVar[Any] = object()

    func: Fn[[T], R]
    payload: R = field(default=SENTINEL, init=False)

    @overload
    def __get__(self, obj: None, obj_type: Optional[type] = ..., /) -> Self: ...
    @overload
    def __get__(self, obj: T, obj_type: Optional[type] = ..., /) -> R: ...
    def __get__(self, obj: T | None, obj_type: Optional[type] = None, /) -> Self | R:
        if obj is None:
            return self
        if self.payload is self.DELETED:
            raise AttributeError("Attribute has been deleted.")
        if self.payload is self.SENTINEL:
            self.payload = self.func(obj)
        return self.payload

    def __set__(self, obj: T, value: R, /) -> None:
        self.payload = value

    def __delete__(self, obj: T, /) -> None:
        self.payload = self.DELETED
