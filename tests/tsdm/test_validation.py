r"""Tests for :mod:`tsdm.testing.validation`."""

import warnings
from hashlib import sha256
from io import BytesIO

import pytest

from tsdm.testing.validation import ValidationError, validate_file_hash


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
