r"""Tests for raw data validation in :mod:`tsdm.datasets.base`."""

from collections.abc import Collection
from functools import cached_property
from inspect import isabstract
from pathlib import Path

import pytest

from tsdm.datasets.base import DatasetBase
from tsdm.testing.validation import ErrorHandler, ValidationError


class DummyDataset(DatasetBase[str, object]):
    r"""Concrete dataset implementation for raw-data validation tests."""

    rawdata_files = []
    table_names = []

    def store_table(self, key: str, table: object, /) -> None:
        pass

    def load_table(self, _key: str, /) -> object:
        return object()


def test_dataset_required_attributes_determine_abstractness() -> None:
    r"""Datasets are abstract until both required attributes are available."""

    class DatasetImplementation(DatasetBase[str, object]):
        def store_table(self, _key: str, _table: object, /) -> None:
            pass

        def load_table(self, _key: str, /) -> object:
            return object()

    class MissingAttributes(DatasetImplementation):
        pass

    class MissingRawdataFiles(DatasetImplementation):
        table_names = ()

    class MissingTableNames(DatasetImplementation):
        rawdata_files = ()

    class AttributeDataset(DatasetImplementation):
        rawdata_files = ()
        table_names = ()

    class PropertyDataset(DatasetImplementation):
        @property
        def rawdata_files(self) -> Collection[str]:  # type: ignore
            return ()

        @property
        def table_names(self) -> Collection[str]:  # type: ignore
            return ()

    class CachedPropertyDataset(DatasetImplementation):
        @cached_property
        def rawdata_files(self) -> Collection[str]:  # type: ignore
            return ()

        @cached_property
        def table_names(self) -> Collection[str]:  # type: ignore
            return ()

    assert isabstract(MissingAttributes)
    assert isabstract(MissingRawdataFiles)
    assert isabstract(MissingTableNames)
    assert not isabstract(AttributeDataset)
    assert not isabstract(PropertyDataset)
    assert not isabstract(CachedPropertyDataset)

    with pytest.raises(
        TypeError,
        match=(
            "abstract class MissingAttributes without an implementation for "
            "abstract methods 'rawdata_files', 'table_names'"
        ),
    ):
        MissingAttributes()
    with pytest.raises(
        TypeError,
        match=(
            "abstract class MissingRawdataFiles without an implementation for "
            "abstract method 'rawdata_files'"
        ),
    ):
        MissingRawdataFiles()
    with pytest.raises(
        TypeError,
        match=(
            "abstract class MissingTableNames without an implementation for "
            "abstract method 'table_names'"
        ),
    ):
        MissingTableNames()


def test_rawdata_without_hash_passes_when_file_exists(tmp_path: Path) -> None:
    r"""A raw file without a reference hash is valid when it exists."""

    class RawDataset(DummyDataset):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]

    dataset = RawDataset(initialize=False)
    dataset.rawdata_paths["raw.bin"].write_bytes(b"raw data")

    assert dataset.validate_rawdata(errors="raise")


def test_rawdata_without_hash_fails_when_file_is_missing(tmp_path: Path) -> None:
    r"""Skipping a hash reference does not skip the existence check."""

    class RawDataset(DummyDataset):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]

    dataset = RawDataset(initialize=False)

    with pytest.raises(ValidationError, match="does not exist"):
        dataset.validate_rawdata(errors="raise")


def test_rawdata_hash_mismatch_raises(tmp_path: Path) -> None:
    r"""Configured raw-data hashes remain strict."""

    class RawDataset(DummyDataset):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]
        rawdata_hashes = {"raw.bin": "sha256:" + "0" * 64}

    dataset = RawDataset(initialize=False)
    dataset.rawdata_paths["raw.bin"].write_bytes(b"raw data")

    with pytest.raises(ValidationError, match="failed validation"):
        dataset.validate_rawdata(errors="raise")


def test_download_uses_custom_rawdata_validation_without_hashes(tmp_path: Path) -> None:
    r"""Custom validation is invoked even when no byte hash is configured."""

    class RawDataset(DummyDataset):
        DATASET_ROOT_DIR = tmp_path
        rawdata_files = ["raw.bin"]
        validation_was_called = False

        def get_rawdata_file(self, fname: str, /) -> None:
            self.rawdata_paths[fname].write_bytes(b"raw data")

        def validate_rawdata_file(
            self, fname: str, /, *, errors: ErrorHandler.Mode = "warn"
        ) -> bool:
            self.validation_was_called = True
            return super().validate_rawdata_file(fname, errors=errors)

    dataset = RawDataset(initialize=False)
    dataset.validation_was_called = False
    dataset.get_rawdata()

    assert dataset.validation_was_called
