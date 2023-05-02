#!/usr/bin/env python

import pandas as pd


class MWE:
    @property
    def foo(self):
        s = pd.Index([1, 2, 3])
        return s.iloc[0]  # actual bug! pd.Index has no .iloc attribute


# MWE().foo  # AttributeError: 'Index' object has no attribute 'iloc'


class MWE_with_getattr:
    def __getattr__(self, key):
        if key == "dummy":
            return "foo"
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}!")

    @property
    def foo(self):
        s = pd.Index([1, 2, 3])
        return s.iloc[0]  # actual bug! pd.Index has no .iloc attribute


MWE_with_getattr().foo  # AttributeError: MWE_with_getattr has no attribute foo!
# Traceback never mentions "AttributeError: 'Index' object has no attribute 'iloc'"


from functools import cached_property


class Foo:
    def __getattribute__(self, item):
        print("__getattribute__ called!")
        try:
            attr = object.__getattribute__(self, item)
        except AttributeError as E:
            try:
                attr = self.__getattr__(item)
            except AttributeError as F:
                F.add_note(str(E))  #
                raise F from E
        return attr

    def __getattr__(self, key):
        if key == "secret":
            return "cake"
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}!")


class Bar(Foo):
    @property
    def prop(self) -> int:
        raise AttributeError("Invisible Error message")

    @cached_property
    def cached_prop(self) -> int:
        raise AttributeError("Invisible Error message")

    filler: str = "Lorem_ipsum"


obj = Bar()
obj.prop  # ✘ AttributeError: Bar has no attribute prop!
obj.cached_prop  # ✘ AttributeError: Bar has no attribute cached_prop!
