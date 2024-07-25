import json
from enum import StrEnum
from typing import Literal, assert_type


class MODE(StrEnum):
    A = "a"
    B = "b"
    C = "c"


with open("data.json", "w") as f:
    json.dump({"mode": MODE.A}, f)


reveal_type(MODE.A)


class Foo[T: MODE]:
    mode: Literal[MODE.A, MODE.B, MODE.C] = MODE.A

    def __init__(self, value: T | str) -> None:
        mode = MODE(value)
        reveal_type(mode)
        self.mode = mode


foo = Foo(MODE.A)
reveal_type(foo.mode)
assert_type(foo.mode, Literal[MODE.A])
