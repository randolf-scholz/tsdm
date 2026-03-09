r"""The KIWI Benchmark Dataset."""

__all__ = ["KiwiBenchmark"]

from zipfile import ZipFile

from pandas import DataFrame

from tsdm.datasets.base import DatasetBase
from tsdm.types.timeseries import TSC_Keys


class KiwiBenchmark(DatasetBase[TSC_Keys, DataFrame]):
    r"""KIWI Benchmark Dataset."""

    __version__: str = "1.0"  # pyright: ignore[reportIncompatibleVariableOverride]

    SOURCE_URL = (
        r"https://tubcloud.tu-berlin.de/s/3CyRJMSqj5feQo2/download?path=%2F&files="
    )
    INFO_URL = r"https://kiwi-biolab.de/"
    HOME_URL = r"https://kiwi-biolab.de/"
    GITHUB_URL = r"https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/kiwi-dataset"

    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "static_covariates",
        "timeseries_metadata",
        "static_covariates_metadata",
    ]
    rawdata_files = ["kiwi-benchmark.zip"]
    rawdata_hashes = {
        "kiwi-benchmark.zip": "sha256:4157b04b348900a20641296b3960a13db44e9098a78737bae2298a9963a217ce"
    }

    def clean_table(self, key: str) -> None:
        path = self.rawdata_paths["kiwi-benchmark.zip"]
        file = f"{key}.parquet"

        with ZipFile(path, "r") as archive:
            try:
                archive.extract(file, self.DATASET_DIR)
            except KeyError as exc:
                exc.add_note(f"Failed to extract table {key} from {path}")
                raise
