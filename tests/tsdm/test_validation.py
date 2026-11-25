r"""Tests for :mod:`tsdm.testing.validation`."""

import csv
import gzip
import warnings
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from os import fdopen, pipe
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from tsdm.validation import (
    ValidationError,
    _infer_schema,
    validate_csv_header,
    validate_file_hash,
    validate_hash,
    validate_table_schema,
)


@pytest.mark.parametrize(
    ("content", "columns", "separator"),
    [
        (b"a,b\n1,2\n", ["a", "b"], ","),
        (b'"a,b","say ""hi"""\r\n1,2\r\n', ["a,b", 'say "hi"'], ","),
        (b'"a\nb";c\n1;2\n', ["a\nb", "c"], ";"),
        ("\ufeffα;β\r1;2\r".encode(), ["α", "β"], ";"),
        (b"a,a,\n", ["a", "a", ""], ","),
    ],
)
def test_validate_csv_header(
    tmp_path: Path, content: bytes, columns: list[str], separator: str
) -> None:
    r"""Paths and binary streams support CSV quoting, UTF-8, and delimiters."""
    path = tmp_path / "table.csv"
    path.write_bytes(content)
    validate_csv_header(path, columns, separator=separator)
    validate_csv_header(str(path), iter(columns), separator=separator)

    with BytesIO(b"prefix\n" + content) as stream:
        stream.seek(7)
        validate_csv_header(stream, columns, separator=separator)
        assert stream.tell() == 7
        assert stream.read() == content


@pytest.mark.parametrize(
    ("content", "columns", "error", "message"),
    [
        (b"a,b\n", ["a", "c"], ValueError, "missing: \\['c'\\]"),
        (b"a,b\n", ["a"], ValueError, "superfluous: \\['b'\\]"),
        (b"a,b\n", ["b", "a"], ValueError, "different orders"),
        (b"a,a\n", ["a"], ValueError, "different numbers"),
        (b"", [], ValueError, "empty"),
        (b'"unterminated\n', ["a"], csv.Error, "unexpected end"),
        (b"\xff\n", ["a"], UnicodeDecodeError, "codec can't decode"),
    ],
)
def test_validate_csv_header_failure(
    content: bytes, columns: list[str], error: type[Exception], message: str
) -> None:
    r"""Failures report mismatches and leave the stream open at its position."""
    with BytesIO(b"prefix\n" + content) as stream:
        stream.seek(7)
        with pytest.raises(error, match=message):
            validate_csv_header(stream, columns)
        assert stream.tell() == 7
        assert stream.read() == content


def test_validate_csv_header_compressed_stream() -> None:
    r"""Compressed binary streams remain usable after header validation."""
    with (
        BytesIO(gzip.compress(b"a,b\n1,2\n")) as compressed,
        gzip.open(compressed, "rb") as stream,
    ):
        validate_csv_header(stream, ["a", "b"])  # pyrefly: ignore[bad-argument-type]
        assert stream.tell() == 0
        assert stream.read() == b"a,b\n1,2\n"


def test_validate_csv_header_non_seekable() -> None:
    r"""Non-seekable streams are rejected without closing or reading them."""
    read_fd, write_fd = pipe()
    with fdopen(read_fd, "rb") as stream, fdopen(write_fd, "wb"):
        with pytest.raises(ValueError, match="non-seekable"):
            validate_csv_header(stream, ["a"])
        assert not stream.closed


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("abcd", "abcd"),
        ("sha256:abcd", "sha256:abcd"),
        ("sha256:abcd", "abcd"),
        ("abcd", "sha256:abcd"),
    ],
)
def test_validate_hash_accepts_matching_digests(actual: str, expected: str) -> None:
    r"""Matching digests pass with missing or matching algorithm labels."""
    assert validate_hash(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    [
        ("sha256:abcd", "sha3_256:abcd", "Hash algorithm mismatch"),
        ("sha256:abcd", "SHA-256:abcd", "Hash algorithm mismatch"),
        ("abcd", "ef01", "Hash mismatch"),
        ("sha256:abcd", "ef01", "Hash mismatch"),
        ("abcd", "sha256:ef01", "Hash mismatch"),
    ],
)
def test_validate_hash_rejects_mismatches(
    actual: str, expected: str, message: str
) -> None:
    r"""A digest mismatch or conflicting algorithm label fails validation."""
    with pytest.raises(ValidationError, match=message):
        validate_hash(actual, expected)
    assert not validate_hash(actual, expected, errors="ignore")


@pytest.mark.parametrize(
    ("prefix", "algorithm"), [("", "sha256"), ("sha256:", "sha256"), ("SHA-256:", None)]
)
def test_validate_file_hash_selects_algorithm(
    prefix: str, algorithm: str | None
) -> None:
    r"""A file checksum algorithm can come from the reference or the caller."""
    payload = b"test payload"
    expected = f"{prefix}{sha256(payload).hexdigest()}"
    with BytesIO(payload) as stream:
        assert validate_file_hash(stream, expected, hash_algorithm=algorithm)


@pytest.mark.parametrize(
    ("reference", "algorithm", "message"),
    [
        ("abcd", None, "Expected hash must specify an algorithm"),
        ("sha3_256:abcd", "sha256", "Hash algorithm mismatch"),
        ("SHA-256:abcd", "sha256", "Hash algorithm mismatch"),
    ],
)
def test_validate_file_hash_checks_algorithm_before_reading(
    tmp_path: Path, reference: str, algorithm: str | None, message: str
) -> None:
    r"""Missing or conflicting algorithms fail before the file is opened."""
    path = tmp_path / "missing.bin"
    with pytest.raises(ValidationError, match=message):
        validate_file_hash(path, reference, hash_algorithm=algorithm)


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
    valid = validate_table_schema(
        table, expected_schema={"label": None, "value": expected_dtype}, errors="raise"
    )
    assert valid


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

    valid = validate_table_schema(
        table, expected_schema=expected_schema, errors="ignore"
    )
    assert not valid
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


@pytest.mark.parametrize(
    ("table", "expected_schema"),
    [
        (
            pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["left", "right"]),
            {"left": pd.Series([1]).dtype, "right": pd.Series([1]).dtype},
        ),
        (pd.Index([1, 2], name="id"), {"id": pd.Series([1]).dtype}),
        (
            pd.Series([1, 2], name="value", index=pd.Index([3, 4], name="id")),
            {"value": pd.Series([1]).dtype},
        ),
        (
            pd.DataFrame({"value": [1, 2]}, index=pd.Index([3, 4], name="id")),
            {"value": pd.Series([1]).dtype},
        ),
        (
            pa.table({"value": [1, 2]}),
            {"value": pa.int64()},
        ),
        (
            pl.DataFrame({"value": [1, 2]}),
            {"value": pl.Int64},
        ),
    ],
)
def test_infer_schema(table, expected_schema) -> None:
    r"""Extract schema information from each supported table type."""
    assert _infer_schema(table) == expected_schema
