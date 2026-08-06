r"""The KIWI Benchmark Dataset."""

__all__ = ["KiwiBenchmark"]

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

    SOURCE_URL = (
        r"https://tubcloud.tu-berlin.de/s/3CyRJMSqj5feQo2/download?path=%2F&files="
    )
    INFO_URL = r"https://kiwi-biolab.de/"
    HOME_URL = r"https://kiwi-biolab.de/"
    GITHUB_URL = r"https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/kiwi-dataset"

    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "timeseries_metadata",
        "static_covariates",
        "static_covariates_metadata",
    ]
    rawdata_files = ["kiwi-benchmark.zip"]
    rawdata_hashes = {
        "kiwi-benchmark.zip": "sha256:4157b04b348900a20641296b3960a13db44e9098a78737bae2298a9963a217ce"
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
