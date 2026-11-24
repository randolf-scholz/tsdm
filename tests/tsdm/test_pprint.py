r"""Test tsdm.utils.pprint."""

import math
from collections.abc import Iterator, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from tsdm.types import SupportsArray
from tsdm.types.dataclass import DataclassType
from tsdm.types.namedtuple import NamedTupleType
from tsdm.utils.pprint import (
    INDENT,
    repr_array,
    repr_compact,
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
def test_pprint_namedtuple(tup: NamedTupleType, expected: str) -> None:
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
def test_pprint_dataclass(dc: DataclassType, expected: str) -> None:
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


DICT_TYPE = pa.dictionary(pa.int32(), pa.string())
TABLE_DATA = {"name": ["foo", "bar"], "value": [0.2, 0.3], "ID": [1, 2]}


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (np.array(1.25)                              , "ndarray(1.25)"                      ),
        (torch.tensor(1.25)                          , "Tensor(1.25)"                       ),
        (np.array([])                                , "ndarray<0>"                         ),
        (np.array([[]])                              , "ndarray<1,0>"                       ),
        (np.empty((0, 0))                            , "ndarray<0,0>"                       ),
        (np.array([1.25])                            , "ndarray<1>(1.25)"                   ),
        (np.array([[1.25]])                          , "ndarray<1,1>(1.25)"                 ),
        (np.array([1, 2, 3])                         , "ndarray<3>[int64]"                  ),
        (np.array([[1.4, 2.3], [3.2, 4.1]])          , "ndarray<2,2>[float64]"              ),
        (torch.tensor([1, 2, 3])                     , "Tensor<3>[int64]"                   ),
        (torch.tensor([[1.4, 2.3], [3.2, 4.1]])      , "Tensor<2,2>[float32]"               ),
        (pd.DataFrame(TABLE_DATA)                    , "DataFrame<2,3>[str, float64, int64]"),
        (pd.Series([1, 2, 3])                        , "Series<3>[int64]"                   ),
        (pd.Series([1, 2, 3], dtype="int32[pyarrow]"), "Series<3>[int32]"                   ),
        (pa.array([1,2,3], type=pa.int16())          , "Int16Array<3>[int16]"               ),
        (pl.Series([1, 2, 3])                        , "Series<3>[Int64]"                   ),
        (pl.Series(["foo", "bar"])                   , "Series<2>[String]"                  ),
        (pl.DataFrame(TABLE_DATA)                    , "DataFrame<2,3>[String, Float64, Int64]"),
        (pa.array(["foo", "bar"], type=DICT_TYPE)    , "DictionaryArray<2>[dict[int32,string]]"),
        (pa.table(TABLE_DATA)                        , "Table<2,3>[STRING8, FLOAT64, INT64]")
    ],
)  # fmt: skip
def test_pprint_array(obj: SupportsArray, expected: str) -> None:
    r"""Test pprint_array."""
    result = repr_array(obj)
    expected = expected.replace("\t", INDENTATION)
    assert result == expected


def test_pprint_lazy_frame() -> None:
    r"""Pretty-print lazy frames without materializing their rows."""
    frame = pl.LazyFrame({"a": [1, 2], "b": ["x", "y"]})
    expected = "LazyFrame<*,2>[Int64, String]"

    assert repr_array(frame) == expected
    assert repr_object(frame) == expected


@pytest.mark.parametrize(
    ("obj", "expected_compact", "expected_object"),
    [
        (np.True_, "True", "bool(True)"),
        (np.False_, "False", "bool(False)"),
        (np.int64(1), "1", "int64(1)"),
        (np.float32(1.25), "1.25", "float32(1.25)"),
        (np.complex64(1 + 2j), "(1+2j)", "complex64((1+2j))"),
    ],
)
def test_pprint_numpy_scalar(
    obj: np.generic, expected_compact: str, expected_object: str
) -> None:
    r"""Check compact and full representations of NumPy scalars."""
    assert repr_compact(obj) == expected_compact
    assert repr_object(obj) == expected_object


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (pd.NA, "NAType()"),  # unfortunate.
        (pd.NaT, "NaT"),
        (math.inf, "inf"),
        (math.nan, "nan"),
        (None, "None"),
    ],
)
def test_print_scalar(obj, expected: str) -> None:
    assert repr_object(obj) == expected
    assert repr_compact(obj) == expected
