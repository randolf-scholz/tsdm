r"""Testing of In Silico dataset, as a token for the whole BaseDataset architecture."""

import logging

from tsdm.datasets import BaseDataset, Dataset, InSilico
from tsdm.utils.decorators import timefun

__logger__ = logging.getLogger(__name__)


def test_caching() -> None:
    """Test the caching of the dataset."""
    # NOTE: this test must be executed first!!!
    LOGGER = __logger__.getChild(InSilico.__name__)
    LOGGER.info("Testing caching.")

    ds = InSilico(initialize=False)
    _, pre_cache_time = timefun(lambda: ds.timeseries, append=True)()
    _, post_cache_time = timefun(lambda: ds.timeseries, append=True)()

    LOGGER.info("%f, %f", pre_cache_time, post_cache_time)

    assert (
        100 * post_cache_time <= pre_cache_time
    ), f"{post_cache_time=}, {pre_cache_time=}"

    LOGGER.info("%s passes caching test ✔.")


def test_dataset_protocol() -> None:
    """Test the attributes of the dataset."""
    LOGGER = __logger__.getChild(InSilico.__name__)
    LOGGER.info("Testing attributes.")
    ds = InSilico()
    assert isinstance(ds, Dataset)
    assert isinstance(ds, BaseDataset)
