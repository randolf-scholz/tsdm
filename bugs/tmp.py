#!/usr/bin/env python3

from types import EllipsisType

x = ...
match x:
    case EllipsisType():
        print(0)
