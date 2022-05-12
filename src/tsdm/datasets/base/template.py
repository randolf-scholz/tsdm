r"""A rough Template how an implemented dataset should look like."""

__all__ = [
    # Classes
    "Template",
]

from functools import cached_property
from pathlib import Path

from tsdm.datasets.base._base import BaseDataset
from tsdm.util.types import NullableNestedType


class Template(BaseDataset):
    r"""A rough Template how an implemented dataset should look like."""

    @property
    def rawdata_files(self) -> NullableNestedType[Path]:
        r"""Location of (possibly compressed) data archive."""
        return self.rawdata_dir / f"{self.__class__.__name__}.csv"

    @cached_property
    def dataset_files(self) -> Path:
        r"""Location of the main data file."""
        return self.dataset_dir / f"{self.__class__.__name__}.h5"

    def clean(self):
        r"""Create DataFrame from raw data."""

    def load(self):
        r"""Load the pre-preprocessed dataset from disk."""
        # df = read_hdf(cls.dataset_files, key=cls.__name__)
        # return DataFrame(df)
