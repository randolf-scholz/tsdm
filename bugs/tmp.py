#!/usr/bin/env python3

from typing import Literal, TypeAlias

Action: TypeAlias = Literal[
    "store",
    "store_const",
    "store_true",
    "append",
    "append_const",
    "count",
    "help",
    "version",
]

x: Literal[0] = NotImplemented
y: float = NotImplemented
