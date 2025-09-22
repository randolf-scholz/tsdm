from dataclasses import dataclass, fields
from typing import ClassVar

import pytest

from tsdm.types.protocols import is_dataclass


class Meta(type):
    @property
    def FIELDS(cls) -> set[str]:
        r"""Return the fields of the class."""
        print("!!!!!")
        if is_dataclass(cls):
            return {f.name for f in fields(cls)}
        return set()


class Base(metaclass=Meta):
    FIELDS: ClassVar[set[str]]
    r"""The fields of the class."""

    def __init_subclass__(cls) -> None:
        print("Inside __init_subclass__")


@dataclass
class Demo(Base):
    x: float
    y: float


def test_fields() -> None:
    assert Base.FIELDS == set()

    assert Demo.FIELDS == {"x", "y"}

    # check that fields are read-only
    with pytest.raises(AttributeError):
        Demo.fields = {}
