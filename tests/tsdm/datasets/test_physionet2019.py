r"""Test PhysioNet 2019."""

from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pytest

from tsdm.datasets import PhysioNet2019
from tsdm.validation import ValidationError


@pytest.fixture
def local_dataset(tmp_path: Path) -> PhysioNet2019:
    r"""Create a dataset with small ZIP archives and known content digests."""

    class LocalPhysioNet2019(PhysioNet2019):
        DATASET_ROOT_DIR = tmp_path
        rawdata_hashes = dict.fromkeys(
            PhysioNet2019.rawdata_files,
            "sha256:14d37467ebb2e374bb28ebd40e2caecf17922c8c40d084d83b57fd16b3fdeb1c",
        )

    dataset = LocalPhysioNet2019(initialize=False, verbose=False)
    for path in dataset.rawdata_paths.values():
        with ZipFile(path, "w") as archive:
            archive.writestr("test.txt", "contents")
    return dataset


def test_validate_rawdata_checks_zip_contents(local_dataset: PhysioNet2019) -> None:
    r"""A changed member fails single-file, bulk, and pre-clean validation."""
    fname = "training_setB.zip"
    assert local_dataset.validate_rawdata(fname)
    assert local_dataset.validate_rawdata()

    with ZipFile(local_dataset.rawdata_paths[fname], "w") as archive:
        archive.writestr("test.txt", "changed contents")

    with pytest.warns(UserWarning, match="Hash mismatch"):
        assert not local_dataset.validate_rawdata(fname)
    assert not local_dataset.validate_rawdata(errors="ignore")
    with pytest.raises(ValidationError, match=r"training_setB\.zip"):
        local_dataset.validate_rawdata(errors="raise")
    with pytest.raises(ValidationError, match=r"training_setB\.zip"):
        local_dataset.clean()


def test_validate_rawdata_reports_missing_archive(local_dataset: PhysioNet2019) -> None:
    r"""A missing archive is reported according to the requested error mode."""
    fname = "training_setA.zip"
    local_dataset.rawdata_paths[fname].unlink()

    assert not local_dataset.validate_rawdata(fname, errors="ignore")
    with pytest.raises(ValidationError, match="Raw data file does not exist"):
        local_dataset.validate_rawdata(fname, errors="raise")


@pytest.mark.manual
def test_physionet_2019_preprocessing() -> None:
    PhysioNet2019.reset_dataset_files(force=True)
    ds = PhysioNet2019()

    assert ds.timeseries_metadata.height == ds.timeseries.width
    assert ds.static_covariates_metadata.height == ds.static_covariates.width

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
