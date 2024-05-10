r"""Test tsdm.utils.pprint."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from typing_extensions import Any, NamedTuple

from tsdm.utils.pprint import (
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_sequence,
)


@dataclass
class ExampleDataclass:
    r"""Example Dataclass."""

    a: int
    b: str
    c: float


class ExampleNamedTuple(NamedTuple):
    r"""Example NamedTuple."""

    a: int
    b: str
    c: float


class ExampleMapping(Mapping):
    r"""Example Mapping."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class ExampleSequence(Sequence):
    r"""Example Sequence."""

    def __init__(self, data: list[Any]):
        self.data = data

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


def test_pprint_namedtuple():
    r"""Test pprint_namedtuple."""
    tup = ExampleNamedTuple(1, "a", 3.14)
    expected = """\
    ExampleNamedTuple<namedtuple>(
        a: 1,
        b: "a",
        c: 3.14,
    )"""
    result = repr_namedtuple(tup)
    assert result == expected


def test_pprint_dataclass():
    r"""Test pprint_dataclass."""
    dc = ExampleDataclass(1, "a", 3.14)
    expected = """\
    ExampleDataclass<dataclass>(
        a: 1,
        b: a,
        c: 3.14,
    )"""
    result = repr_dataclass(dc)
    assert result == expected


def test_pprint_mapping():
    r"""Test pprint_mapping."""
    mp = ExampleMapping({"a": 1, "b": "a", "c": 3.14})
    expected = """\
    ExampleMapping<mapping>(
        a: 1,
        b: "a",
        c: 3.14,
    )"""
    result = repr_mapping(mp)
    assert result == expected


def test_pprint_sequence():
    r"""Test pprint_sequence."""
    seq = ExampleSequence([1, "a", 3.14])
    expected = """\
    ExampleSequence<sequence>(
        1,
        "a",
        3.14,
    )"""
    result = repr_sequence(seq)
    assert result == expected
