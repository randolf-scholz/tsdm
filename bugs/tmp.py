#!/usr/bin/env python3

from typing import Callable, Concatenate, ParamSpec, TypeAlias

P = ParamSpec("P")

intfun: TypeAlias = Callable[Concatenate[int, ..., P], None]  # ✘
