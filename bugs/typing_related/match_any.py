#!/usr/bin/env python

from typing import Any


class UntypedClass(Any):
    """Dummy class."""


reveal_type(UntypedClass)


match 123:
    case UntypedClass():
        print("UntypedClass")
    case _:
        print("other")
