r"""Test tsdm.utils.pprint."""

import math
from collections.abc import Iterator, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from tsdm.types import HasDTypes, SupportsArray, SupportsDataFrame
from tsdm.types.dataclass import DataclassType
from tsdm.types.namedtuple import NamedTupleType
from tsdm.utils.pprint import (
    INDENT,
    Identifier,
    _repr_array,
    _repr_dataclass,
    _repr_mapping,
    _repr_namedtuple,
    _repr_sequence,
    _repr_set,
    _repr_table,
    repr_compact,
    repr_object,
)

INDENTATION = " " * INDENT


def test_identifier_lookup() -> None:
    r"""Infer identifiers from object types."""
    assert Identifier("mapping") is Identifier.MAPPING
    assert Identifier.get("<mapping>") is None
    assert Identifier.get("custom label") is None
    assert Identifier.get([1]) is Identifier.SEQUENCE
    assert Identifier.get(object()) is None
    assert repr_compact(object(), recursive=True) == "object()"
    with pytest.raises(ValueError, match="not a valid Identifier"):
        Identifier("<mapping>")


def test_pprint_types_and_functions() -> None:
    r"""Use shared formatters for types and both kinds of functions."""

    def example() -> None:
        pass

    assert Identifier.get(example) is Identifier.FUNCTION
    assert Identifier.get(len) is Identifier.FUNCTION
    for formatter in (repr_compact, repr_object):
        assert formatter(int, identifier="custom") == "int<custom>"
        assert formatter(example) == "example"
        assert formatter(example, identifier="custom") == "example<custom>"
        assert formatter(len) == "<built-in function len>"
        assert formatter(len, identifier="custom") == "<custom><built-in function len>"


def test_sequence_identifier_excludes_strings() -> None:
    r"""Use the sequence identifier for containers, excluding strings."""
    Identifier.SEQUENCE.validate([1])
    with pytest.raises(AssertionError, match="excludes strings"):
        Identifier.SEQUENCE.validate("text")


def test_identifier_overlapping_categories() -> None:
    r"""Validate categories independently of default dispatch order."""

    @dataclass
    class MappingDataclass(dict):
        r"""A mapping that also matches the dataclass protocol."""

    obj = MappingDataclass()
    assert Identifier.get(obj) is Identifier.DATACLASS
    Identifier.MAPPING.validate(obj)


@pytest.mark.parametrize(
    ("identifier", "label"),
    [("custom", "<custom>"), ("déjà_vu", "<déjà_vu>"), ("", ""), (None, "")],
)
def test_custom_identifier(identifier: str | None, label: str) -> None:
    r"""Display custom identifiers with angle brackets and omit empty labels."""
    assert repr_object(1, identifier=identifier) == f"{label}1"
    assert repr_object([1], identifier=identifier, linebreaks=False) == f"{label}[1]"
    assert repr_object(object(), identifier=identifier) == f"object{label}()"
    assert repr_compact(object(), identifier=identifier) == f"object{label}()"
    assert (
        repr_object(object(), identifier=identifier, fallback=lambda _: "fallback")
        == f"{label}fallback"
    )


@pytest.mark.parametrize("identifier", ["<custom>", "custom label", "2,3", "<>"])
def test_invalid_custom_identifier(identifier: str) -> None:
    r"""Reject labels that fail the Python identifier syntax."""
    with pytest.raises(ValueError, match="Invalid identifier"):
        repr_object([1], identifier=identifier)


@dataclass
class ExampleDataclass:
    r"""Example Dataclass."""

    a: int
    b: str
    c: float


def test_identifier_for_wrapped_payload() -> None:
    r"""Select the formatter from the payload and the label from the wrapper."""
    wrapper = ExampleDataclass(2, "ignored", 3.0)
    result = repr_object([1], wrapped=wrapper, linebreaks=False)
    assert result == "ExampleDataclass<dataclass>(1)"


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
    result = _repr_namedtuple(tup)
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
    result = _repr_dataclass(dc)
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
    result = _repr_mapping(mapping)
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
    result = _repr_sequence(seq)
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
    result = _repr_set(obj)
    expected = {exp.replace("\t", INDENTATION) for exp in expected}
    assert result in expected


DICT_TYPE = pa.dictionary(pa.int32(), pa.string())
TABLE_DATA = {"name": ["foo", "bar"], "value": [0.2, 0.3], "ID": [1, 2]}
ARROW_TABLE = pa.table(TABLE_DATA)


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
        (pd.Series([1, 2, 3])                        , "Series<3>[int64]"                   ),
        (pd.Series([1, 2, 3], dtype="int32[pyarrow]"), "Series<3>[int32]"                   ),
        (pa.array([1,2,3], type=pa.int16())          , "Int16Array<3>[int16]"               ),
        (pl.Series([1, 2, 3])                        , "Series<3>[Int64]"                   ),
        (pl.Series(["foo", "bar"])                   , "Series<2>[String]"                  ),
        (pa.array(["foo", "bar"], type=DICT_TYPE)    , "DictionaryArray<2>[dict[int32,string]]"),
    ],
)  # fmt: skip
def test_pprint_array(obj: SupportsArray, expected: str) -> None:
    r"""Test pprint_array."""
    result = _repr_array(obj)
    expected = expected.replace("\t", INDENTATION)
    assert result == expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (pd.DataFrame(TABLE_DATA), "DataFrame<2,3>[str, float64, int64]"),
        (pl.DataFrame(TABLE_DATA), "DataFrame<2,3>[String, Float64, Int64]"),
        (pl.LazyFrame(TABLE_DATA), "LazyFrame<*,3>[String, Float64, Int64]"),
        (ARROW_TABLE, "Table<2,3>[STRING8, FLOAT64, INT64]"),
        (
            SimpleNamespace(__dataframe__=ARROW_TABLE.__dataframe__),
            "SimpleNamespace<*,3>[STRING8, FLOAT64, INT64]",
        ),
        (
            SimpleNamespace(dtypes=pd.DataFrame(TABLE_DATA).dtypes),
            "SimpleNamespace<*,3>[str, float64, int64]",
        ),
        (pa.table({"value": [1]}), "Table<1,1>[INT64]"),
    ],
)
def test_pprint_table(obj: SupportsDataFrame | HasDTypes, expected: str) -> None:
    r"""Format the same data across the supported table interfaces."""
    assert Identifier.get(obj) is Identifier.TABLE
    assert _repr_table(obj) == expected
    assert repr_object(obj) == expected


def test_pprint_table_row_count_from_length() -> None:
    r"""Use the table length as the row count when there is no shape."""

    class SizedTable:
        r"""A table exposing column dtypes and a row count."""

        dtypes = ("int64", "float64")

        def __len__(self) -> int:
            return 3

    assert repr_object(SizedTable()) == "SizedTable<3,2>[int64, float64]"


def test_pprint_lazy_frame_identifier() -> None:
    r"""Display a custom identifier for a lazy frame."""
    frame = pl.LazyFrame(TABLE_DATA)
    assert (
        repr_object(frame, identifier="sample")
        == "LazyFrame<sample>[String, Float64, Int64]"
    )


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
