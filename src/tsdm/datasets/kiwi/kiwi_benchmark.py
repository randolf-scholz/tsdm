r"""The KIWI Benchmark Dataset."""

__all__ = ["KiwiBenchmark"]

from zipfile import ZipFile

from tsdm.datasets.base import MultiTableDataset


class KiwiBenchmark(MultiTableDataset):
    r"""KIWI Benchmark Dataset."""

    SOURCE_URL = r"https://tubcloud.tu-berlin.de/s/YA65b8iieQoWQTW/download/"
    INFO_URL = r"https://kiwi-biolab.de/"
    HOME_URL = r"https://kiwi-biolab.de/"
    GITHUB_URL = r"https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/kiwi-dataset"

    __version__ = "1.0"

    rawdata_files = ["kiwi-benchmark.zip"]
    rawdata_hashes = {
        "kiwi-benchmark.zip": "sha256:dd5eb62dccd5fb7774e7600145fd838f92d55eb07d6b89510c3fddbfd295f928"
    }
    table_names = [
        "timeseries",
        "static_covariates",
        "timeseries_metadata",
        "static_covariates_metadata",
    ]

    def clean_table(self, key: str) -> None:
        path = self.rawdata_paths["kiwi-benchmark.zip"]
        file = f"{key}.parquet"

        with ZipFile(path, "r") as archive:
            try:
                archive.extract(file, self.DATASET_DIR)
            except KeyError as exc:
                exc.add_note(f"Failed to extract table {key} from {path}")
                raise
