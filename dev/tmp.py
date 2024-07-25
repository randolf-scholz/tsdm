#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Literal, TypeAlias

# type Method = Literal["RK45", "RK23"]
Method: TypeAlias = Literal["RK45", "RK23"]


@dataclass
class Solver:
    method: Method
