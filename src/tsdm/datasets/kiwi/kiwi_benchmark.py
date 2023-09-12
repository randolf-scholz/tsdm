"""The KIWI Benchmark Dataset."""

__all__ = ["KIWI_Dataset"]

from zipfile import ZipFile

from tsdm.datasets.base import MultiTableDataset


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


def deprecated(
    func=None, msg=None, /, *, category=DeprecationWarning, stacklevel: int = 1
):
    """Indicate that a class, function or overload is deprecated."""
    if isinstance(func, str):
        # used as deprecated("message") -> shift arguments
        assert msg is None
        msg = func
        func = None

    if func is None:
        # used with brackets -> decorator factory
        def decorator(decorated):
            msg = make_default_message(decorated) if msg is None else msg

            def wrapped(*args, **kwargs):
                ...

            return wrapped

        return decorator

    # used without brackets -> wrap func
    msg = make_default_message(func)

    def wrapped(*args, **kwargs):
        ...

    return wrapped
