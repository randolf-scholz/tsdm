r"""Tests for raw data validation in :mod:`tsdm.datasets.base`."""

from collections.abc import Collection
from pathlib import Path

import pytest

from tsdm.datasets.base import DatasetBase
from tsdm.testing.validation import ErrorHandler, ValidationError


def test_rawdata_without_hash_passes_when_file_exists(tmp_path: Path) -> None:
    r"""A raw file without a reference hash is valid when it exists."""

    class RawDataset(DatasetBase[str, object]):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]
        table_names: Collection[str] = ()

    dataset = RawDataset(initialize=False)
    dataset.rawdata_paths["raw.bin"].write_bytes(b"raw data")

    assert dataset.validate_rawdata(errors="raise")


def test_rawdata_without_hash_fails_when_file_is_missing(tmp_path: Path) -> None:
    r"""Skipping a hash reference does not skip the existence check."""

    class RawDataset(DatasetBase[str, object]):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]
        table_names: Collection[str] = ()

    dataset = RawDataset(initialize=False)

    with pytest.raises(ValidationError, match="does not exist"):
        dataset.validate_rawdata(errors="raise")


def test_rawdata_hash_mismatch_raises(tmp_path: Path) -> None:
    r"""Configured raw-data hashes remain strict."""

    class RawDataset(DatasetBase[str, object]):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]
        rawdata_hashes = {"raw.bin": "sha256:" + "0" * 64}
        table_names: Collection[str] = ()

    dataset = RawDataset(initialize=False)
    dataset.rawdata_paths["raw.bin"].write_bytes(b"raw data")

    with pytest.raises(ValidationError, match="failed validation"):
        dataset.validate_rawdata(errors="raise")


def test_download_uses_custom_rawdata_validation_without_hashes(tmp_path: Path) -> None:
    r"""Custom validation is invoked even when no byte hash is configured."""

    class RawDataset(DatasetBase[str, object]):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]
        table_names: Collection[str] = ()
        validation_was_called = False

        def get_rawdata_file(self, fname: str, /) -> None:
            self.rawdata_paths[fname].write_bytes(b"raw data")

        def validate_rawdata_file(
            self, key: str, /, *, errors: ErrorHandler.Mode = "warn"
        ) -> bool:
            self.validation_was_called = True
            return super().validate_rawdata_file(key, errors=errors)

    dataset = RawDataset(initialize=False)
    dataset.validation_was_called = False
    dataset.get_rawdata()

    assert dataset.validation_was_called
