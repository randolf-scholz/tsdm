r"""A rough Template how an implemented dataset should look like."""

__all__ = [
    # Classes
    "Template",
]

import logging
from functools import cached_property
from pathlib import Path

from tsdm.datasets.base._base import BaseDataset
from tsdm.util.types import NullableNestedType

__logger__ = logging.getLogger(__name__)


class Template(BaseDataset):
    r"""A rough Template how an implemented dataset should look like."""

    @property
    def rawdata_files(self) -> NullableNestedType[Path]:
        r"""Location of (possibly compressed) data archive."""
        return self.rawdata_dir.joinpath(f"{self.name}.csv")

    @cached_property
    def dataset_files(self) -> Path:
        r"""Location of the main data file."""
        return self.dataset_dir.joinpath(f"{self.name}.h5")

    def clean(self):
        r"""Create DataFrame from raw data."""
        __logger__.info("Cleaning dataset '%s'", self.name)
        # df = read_csv(cls.rawdata_paths, **options)
        __logger__.info("Finished extracting dataset '%s'", self.name)
        # df.to_hdf(cls.dataset_files, key=cls.__name__)
        __logger__.info("Finished cleaning dataset '%s'", self.name)

    def load(self):
        r"""Load the pre-preprocessed dataset from disk."""
        # df = read_hdf(cls.dataset_files, key=cls.__name__)
        # return DataFrame(df)
