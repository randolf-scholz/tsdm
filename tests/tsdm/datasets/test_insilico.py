#!/usr/bin/env python
r"""Testing of Electricity dataset, as a token for the whole BaseDataset architecture."""

import logging
from copy import copy

from pandas import DataFrame

from tsdm.datasets import BaseDataset, InSilicoData
from tsdm.utils.decorators import timefun

logging.basicConfig(level=logging.INFO)

__logger__ = logging.getLogger(__name__)


def test_caching():
    """Test the caching of the dataset."""
    # NOTE: this test must be executed first!!!
    __logger__.info("Testing caching of %s.", InSilicoData)
    ds = InSilicoData(initialize=False)
    __logger__.info("Testing caching of dataset %s", ds.__class__.__name__)
    _, pre_cache_time = timefun(lambda: ds.dataset, append=True)()
    _, post_cache_time = timefun(lambda: ds.dataset, append=True)()

    __logger__.info("%f, %f", pre_cache_time, post_cache_time)

    assert (
        100 * post_cache_time <= pre_cache_time
    ), f"{post_cache_time=}, {pre_cache_time=}"

    __logger__.info(
        "%s passes caching test \N{HEAVY CHECK MARK}", ds.__class__.__name__
    )


def test_attributes():
    """Test the attributes of the dataset."""
    __logger__.info("Testing attributes of %s.", InSilicoData)
    ds = InSilicoData()
    base_attrs = copy(set(dir(ds)))
    attrs = {
        "BASE_URL",
        "DATASET_DIR",
        "RAWDATA_DIR",
        "__dir__",
        "clean",
        "dataset",
        "dataset_files",
        "download",
        "load",
    }

    assert attrs <= base_attrs, f"{attrs - base_attrs} missing!"
    assert isinstance(ds.dataset, DataFrame)
    assert issubclass(InSilicoData, BaseDataset)

    for attr in attrs:
        assert hasattr(ds, attr), f"{attr} missing!"


def _main() -> None:
    test_caching()
    test_attributes()


if __name__ == "__main__":
    _main()
