r"""Test class properties defined via metaclass."""

from dataclasses import dataclass, fields
from typing import ClassVar

import pytest

from tsdm.types.dataclass import is_dataclass


class Meta(type):
    @property
    def FIELDS(cls) -> set[str]:  # ruff: ignore[N802]
        r"""Return the fields of the class."""
        if is_dataclass(cls):
            return {f.name for f in fields(cls)}
        return set()


def test_fields() -> None:
    class Base(metaclass=Meta):
        FIELDS: ClassVar[set[str]]
        r"""The fields of the class."""

        def __init_subclass__(cls) -> None:
            print("Inside __init_subclass__")

    @dataclass
    class Demo(Base):
        x: float
        y: float

    assert Base.FIELDS == set()
    assert Demo.FIELDS == {"x", "y"}

    # check that fields are read-only
    with pytest.raises(AttributeError):
        Demo.FIELDS = {}  # type: ignore
