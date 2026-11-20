r"""Tests for :mod:`tsdm.testing.validation`."""

import warnings
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from tsdm.validation import (
    ValidationError,
    validate_file_hash,
    validate_table_schema,
)


def test_validate_file_hash_stream_restores_position() -> None:
    r"""Binary streams are hashed from the beginning and left unchanged."""
    stream = BytesIO(b"test payload")
    stream.seek(4)
    expected_hash = f"sha256:{sha256(stream.getvalue()).hexdigest()}"

    assert validate_file_hash(stream, expected_hash)
    assert stream.tell() == 4


def test_validate_file_hash_stream_restores_position_on_failure() -> None:
    r"""Binary streams are restored even when validation fails."""
    stream = BytesIO(b"test payload")
    stream.seek(4)

    with pytest.raises(ValidationError):
        validate_file_hash(stream, "sha256:deadbeef")

    assert stream.tell() == 4


def test_validate_file_hash_skips_missing_reference(tmp_path) -> None:
    r"""Skipping a missing reference neither hashes nor warns."""
    path = tmp_path / "missing.bin"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate_file_hash(path, None, skipif_no_reference=True)


@pytest.mark.parametrize(
    ("table", "expected_dtype"),
    [
        (pd.DataFrame({"value": [1], "label": ["a"]}), "int64"),
        (pl.DataFrame({"value": [1], "label": ["a"]}), pl.Int64),
        (pa.table({"value": [1], "label": ["a"]}), pa.int64()),
    ],
)
def test_validate_table_schema(table: Any, expected_dtype: Any) -> None:
    r"""Schema order is ignored and `None` skips only its column's dtype check."""
    assert validate_table_schema(
        table, expected_schema={"label": None, "value": expected_dtype}, errors="raise"
    )


@pytest.mark.parametrize(
    ("expected_schema", "message"),
    [
        ({"value": None, "missing": None}, "Missing columns.*"),
        ({}, "Superfluous columns.*"),
        ({"value": "float64"}, "Column 'value'.*"),
        ({"value": None, "id": None}, "Missing columns: \\['id'\\]"),
    ],
)
def test_validate_table_schema_mismatch(
    expected_schema: Mapping[str, Any | None], message: str
) -> None:
    r"""Missing names, extra names, and checked dtype mismatches fail validation."""
    table = pd.DataFrame({"value": [1]}, index=pd.Index([2], name="id"))

    assert not validate_table_schema(
        table, expected_schema=expected_schema, errors="ignore"
    )
    with pytest.warns(UserWarning, match=message):
        assert not validate_table_schema(table, expected_schema=expected_schema)
    with pytest.raises(ValidationError, match=message):
        validate_table_schema(table, expected_schema=expected_schema, errors="raise")


def test_validate_table_schema_ignores_index() -> None:
    r"""A named index is excluded from column validation."""
    table = pd.DataFrame({"value": [1]}, index=pd.Index([2], name="id"))
    assert validate_table_schema(table, expected_schema={"value": None}, errors="raise")


def test_validate_table_schema_empty() -> None:
    r"""An empty schema requires a table without columns."""
    assert validate_table_schema(pd.DataFrame(), expected_schema={}, errors="raise")
