"""The KIWI Benchmark Dataset."""

__all__ = ["KIWI", "KIWI_Dataset"]

from zipfile import ZipFile

from tsdm.datasets.base import MultiTableDataset
from tsdm.datasets.timeseries import TimeSeriesCollection


class KIWI_Dataset(MultiTableDataset):
    r"""KIWI Benchmark Dataset."""

    BASE_URL = r"https://tubcloud.tu-berlin.de/s/rorBS7Lwbgmreti/download/"
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
        "metadata",
        "timeseries_description",
        "metadata_description",
    ]

    def clean_table(self, key: str) -> None:
        with ZipFile(self.rawdata_paths["kiwi-benchmark.zip"], "r") as archive:
            archive.extract(f"{key}.parquet", self.DATASET_DIR)


class KIWI(TimeSeriesCollection):
    r"""The KIWI dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        super().__init__(**KIWI_Dataset())
