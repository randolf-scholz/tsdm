r"""The KIWI Benchmark Dataset."""

__all__ = ["KiwiBenchmark"]

import shutil
from importlib import resources
from typing import Literal
from zipfile import ZipFile

from pandas import DataFrame

from tsdm.datasets.base import DatasetBase

type Key = Literal[
    "timeseries",
    "timeseries_metadata",
    "static_covariates",
    "static_covariates_metadata",
]


class KiwiBenchmark(DatasetBase[Key, DataFrame]):
    r"""KIWI Benchmark Dataset."""

    DEFAULT_VERSION = "1.0"
    __version__: str  # pyright: ignore[reportIncompatibleMethodOverride]

    INFO_URL = r"https://www.tu.berlin/bioprocess/einrichtungen-associates/arbeitsgruppen/kiwi-biolab"
    HOME_URL = r"https://www.tu.berlin/bioprocess/einrichtungen-associates/arbeitsgruppen/kiwi-biolab"

    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "timeseries_metadata",
        "static_covariates",
        "static_covariates_metadata",
    ]
    rawdata_files = ["kiwi-benchmark.zip"]
    rawdata_hashes = {
        "kiwi-benchmark.zip": "sha256:f5af65bbf922aa21bb05b1e3adf68579dc53db8ccf12862fa2a575fbe6fc2652"
    }

    def clean_table(self, key: Key) -> None:
        path = self.rawdata_paths["kiwi-benchmark.zip"]
        file = f"{key}.parquet"

        with ZipFile(path, "r") as archive:
            try:
                archive.extract(file, self.DATASET_DIR)
            except KeyError as exc:
                exc.add_note(f"Failed to extract table {key} from {path}")
                raise

    def download_file(self, fname: str, /) -> None:
        r"""Copy the bundled dataset archive to the raw-data directory."""
        self.LOGGER.info("Copying data files into %s.", self.rawdata_paths[fname])
        if __package__ is None:
            raise ValueError(f"Unexpected package: {__package__=}")
        with resources.path(__package__, fname) as path:
            shutil.copy(path, self.rawdata_paths[fname])
