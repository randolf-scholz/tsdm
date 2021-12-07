r"""Testing of Electricity dataset, as a token for the whole BaseDataset architecture."""

import logging
from copy import copy

from pandas import DataFrame

from tsdm.datasets import BaseDataset, InSilicoData
from tsdm.util.decorators import timefun

LOGGER = logging.getLogger(__name__)


def test_caching():
    """Check if dataset caching works (should be way faster)."""
    # NOTE: this test must be executed first!!!

    ds = InSilicoData()
    LOGGER.info("Testing caching of dataset %s", ds.name)
    _, pre_cache_time = timefun(lambda: ds.dataset)()
    _, post_cache_time = timefun(lambda: ds.dataset)()

    LOGGER.info("%f, %f", pre_cache_time, post_cache_time)

    assert (
        100 * post_cache_time <= pre_cache_time
    ), f"{post_cache_time=}, {pre_cache_time=}"

    LOGGER.info("%s passes caching test \N{HEAVY CHECK MARK}", ds.name)


def test_attributes():
    r"""Test if all attributes are present."""
    ds = InSilicoData()
    base_attrs = copy(set(dir(ds)))
    attrs = {
        "clean",
        "dataset",
        "dataset_files",
        "dataset_dir",
        "download",
        "load",
        "rawdata_dir",
        "base_url",
    }

    assert attrs <= base_attrs, f"{attrs - base_attrs} missing!"
    assert isinstance(ds.dataset, DataFrame)
    assert issubclass(InSilicoData, BaseDataset)
    # assert isinstance(InSilicoData(), BaseDataset)

    assert hasattr(ds, "dataset")
    assert hasattr(ds, "load")
    assert hasattr(ds, "download")
    assert hasattr(ds, "clean")


def __main__():
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Testing CACHING started!")
    test_caching()
    LOGGER.info("Testing CACHING finished!")

    LOGGER.info("Testing ATTRIBUTES started!")
    test_attributes()
    LOGGER.info("Testing ATTRIBUTES finished!")


if __name__ == "__main__":
    __main__()
