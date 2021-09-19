r"""Testing of Electricity dataset, as a token for the whole BaseDataset architecture."""

import logging
from copy import copy

from tsdm.datasets import BaseDataset, InSilicoData
from tsdm.util.decorators import timefun

LOGGER = logging.getLogger(__name__)


def test_caching():
    """Check if dataset caching works (should be way faster)."""
    # NOTE: this test must be executed first!!!

    ds = InSilicoData
    LOGGER.info("Testing caching of dataset %s", ds.__name__)
    _, pre_cache_time = timefun(lambda: ds.dataset)()
    _, post_cache_time = timefun(lambda: ds.dataset)()

    LOGGER.info("%f, %f", pre_cache_time, post_cache_time)

    assert 1000 * post_cache_time <= pre_cache_time

    LOGGER.info("%s passes caching test \N{HEAVY CHECK MARK}", ds.__name__)


def test_attributes():
    r"""Test if all attributes are present."""
    ds = InSilicoData
    base_attrs = copy(set(dir(ds)))
    attrs = {
        "clean",
        "dataset",
        "dataset_file",
        "dataset_path",
        "download",
        "load",
        "rawdata_path",
        "url",
    }

    assert attrs <= base_attrs, f"{attrs - base_attrs} missing!"
    assert isinstance(ds.dataset, dict)
    assert issubclass(InSilicoData, BaseDataset)
    # assert isinstance(InSilicoData(), BaseDataset)

    assert hasattr(InSilicoData, "dataset")
    assert hasattr(InSilicoData, "load")
    assert hasattr(InSilicoData, "download")
    assert hasattr(InSilicoData, "clean")


if __name__ == "__main__":
    test_caching()
    test_attributes()
