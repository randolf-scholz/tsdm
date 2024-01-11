#!/usr/bin/env python

from abc import abstractmethod
from typing import *


class Foo(Protocol):

    @abstractmethod
    def bar(self): ...


class MyFoo(Foo):
    ...


print(MyFoo())
