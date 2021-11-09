r"""A rough Template how an implemented dataset should look like."""

from __future__ import annotations

__all__ = [
    # Classes
    "Template",
]

import logging
from functools import cache
from pathlib import Path

from tsdm.datasets.base.dataset import BaseDataset

__logger__ = logging.getLogger(__name__)


class Template(BaseDataset):
    r"""A rough Template how an implemented dataset should look like."""

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def rawdata_file(cls) -> Path:
        r"""Location of (possibly compressed) data archive."""
        return cls.rawdata_path.joinpath(f"{cls.__name__}.csv")  # type: ignore

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def dataset_file(cls) -> Path:
        r"""Location of the main data file."""
        return cls.dataset_path.joinpath(f"{cls.__name__}.h5")  # type: ignore

    @classmethod
    def clean(cls):
        r"""Create DataFrame from raw data."""
        __logger__.info("Cleaning dataset '%s'", cls.__name__)
        # df = read_csv(cls.rawdata_file, **options)
        __logger__.info("Finished extracting dataset '%s'", cls.__name__)
        # df.to_hdf(cls.dataset_file, key=cls.__name__)
        __logger__.info("Finished cleaning dataset '%s'", cls.__name__)

    @classmethod
    def load(cls):
        r"""Load the pre-preprocessed dataset from disk."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        # df = read_hdf(cls.dataset_file, key=cls.__name__)
        # return DataFrame(df)
