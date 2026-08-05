r"""Test interactions between custom boolean class and numpy ndarray."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SupportsBool(Protocol):
    def __bool__(self) -> bool: ...


@dataclass
class MyBool:
    b: bool

    def __bool__(self) -> bool:
        return bool(self.b)

    def __and__(self, other: SupportsBool) -> MyBool:
        try:
            other_bool = bool(other)
        except Exception:
            return NotImplemented
        return MyBool(bool(self) & other_bool)

    def __rand__(self, other: SupportsBool) -> MyBool:
        try:
            other_bool = bool(other)
        except Exception:
            return NotImplemented
        return MyBool(bool(self) & other_bool)


def test_and_ndarray() -> None:
    a = MyBool(True)  # noqa: FBT003
    b = np.array([True, False, True])
    # the values of result are MyBool, not builtins.bool:
    assert all(isinstance(x, MyBool) for x in a & b)  # type: ignore
    assert all(isinstance(x, MyBool) for x in b & a)
