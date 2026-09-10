r"""Test dataclass fields do not include properties."""

from dataclasses import dataclass, fields


def test_property_field() -> None:
    @dataclass
    class Demo:
        # value: Final[str]

        def __init__(self) -> None:
            pass

        @property
        def value(self) -> str:
            return "abc"

    obj = Demo()
    assert obj.value == "abc"

    cls_fields = {field.name for field in fields(Demo)}
    assert "value" not in cls_fields

    obj_fields = {field.name for field in fields(obj)}
    assert "value" not in obj_fields
