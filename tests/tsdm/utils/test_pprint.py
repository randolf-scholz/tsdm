r"""Test tsdm.utils.pprint."""

from collections.abc import Iterator, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from typing_extensions import Any, NamedTuple

from tsdm.types.protocols import Dataclass, NTuple, SupportsArray
from tsdm.utils.pprint import (
    INDENT,
    repr_array,
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_object,
    repr_sequence,
    repr_set,
)

INDENTATION = " " * INDENT


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

    def __getitem__(self, index: int | slice) -> Any:
        return self.data[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (1, "1"),
        (1.25, "1.25"),
        (1.0 + 2.0j, "(1+2j)"),
        ("a", "'a'"),
        (None, "None"),
        (True, "True"),
        (False, "False"),
        (Ellipsis, "Ellipsis"),
        (NotImplemented, "NotImplemented"),
    ],
)
def test_pprint_object(obj: object, expected: str) -> None:
    r"""Test pprint_object."""
    result = repr_object(obj)
    assert result == expected


@pytest.mark.parametrize(
    ("tup", "expected"),
    [
        (
            ExampleNamedTuple(1, "a", 1.25),
            "ExampleNamedTuple<namedtuple>(\n\ta: 1,\n\tb: 'a',\n\tc: 1.25,\n)",
        ),
    ],
)
def test_pprint_namedtuple(tup: NTuple, expected: str) -> None:
    r"""Test pprint_namedtuple."""
    result = repr_namedtuple(tup)
    expected = expected.replace("\t", INDENTATION)
    assert result == expected


@pytest.mark.parametrize(
    ("dc", "expected"),
    [
        (
            ExampleDataclass(1, "a", 1.25),
            "ExampleDataclass<dataclass>(\n\ta: 1,\n\tb: 'a',\n\tc: 1.25,\n)",
        ),
    ],
)
def test_pprint_dataclass(dc: Dataclass, expected: str) -> None:
    r"""Test pprint_dataclass."""
    result = repr_dataclass(dc)
    expected = expected.replace("\t", INDENTATION)
    assert result == expected


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (
            {"foo": 0, "foobar": 1},
            "{\n\tfoo   : 0,\n\tfoobar: 1,\n}",
        ),
        (
            ExampleMapping({"a": 1, "b": "a", "c": 1.25}),
            "ExampleMapping<mapping>(\n\ta: 1,\n\tb: 'a',\n\tc: 1.25,\n)",
        ),
    ],
)
def test_pprint_mapping(mapping: Mapping, expected: str) -> None:
    r"""Test pprint_mapping."""
    result = repr_mapping(mapping)
    expected = expected.replace("\t", INDENTATION)
    assert result == expected


@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        (
            (1, "a", 1.25),
            "(\n\t1,\n\t'a',\n\t1.25,\n)",
        ),
        (
            [1, "a", 1.25],
            "[\n\t1,\n\t'a',\n\t1.25,\n]",
        ),
        (
            ExampleSequence([1, "a", 1.25]),
            "ExampleSequence<sequence>(\n\t1,\n\t'a',\n\t1.25,\n)",
        ),
    ],
)
def test_pprint_sequence(seq: Sequence, expected: str) -> None:
    r"""Test pprint_sequence."""
    result = repr_sequence(seq)
    expected = expected.replace("\t", INDENTATION)
    assert result == expected


# NOTE: Order is not guaranteed for sets.
@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (
            {1, "a"},
            {
                "{\n\t1,\n\t'a',\n}",
                "{\n\t'a',\n\t1,\n}",
            },
        ),
        (
            frozenset({1, "a"}),
            {
                "frozenset(\n\t1,\n\t'a',\n)",
                "frozenset(\n\t'a',\n\t1,\n)",
            },
        ),
    ],
)
def test_pprint_set(obj: AbstractSet, expected: set[str]) -> None:
    r"""Test pprint_set."""
    result = repr_set(obj)
    expected = {exp.replace("\t", INDENTATION) for exp in expected}
    assert result in expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (
            np.array(1.25),
            "ndarray[1.25, float64]",
        ),
        (
            torch.tensor(1.25),
            "Tensor[1.25, torch.float32@cpu]",
        ),
        (
            np.array([1, 2, 3]),
            "ndarray[(3,), int64]",
        ),
        (
            torch.tensor([1, 2, 3]),
            "Tensor[(3,), torch.int64@cpu]",
        ),
        (
            np.array([[1.4, 2.3], [3.2, 4.1]]),
            "ndarray[(2, 2), float64]",
        ),
        (
            torch.tensor([[1.4, 2.3], [3.2, 4.1]]),
            "Tensor[(2, 2), torch.float32@cpu]",
        ),
    ],
)
def test_pprint_array(obj: SupportsArray, expected: str) -> None:
    r"""Test pprint_array."""
    result = repr_array(obj)
    expected = expected.replace("\t", INDENTATION)
    assert result == expected
