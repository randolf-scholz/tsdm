"""
Testing of Electricity dataset, as a token for the whole BaseDataset architecture.
"""

import logging
from copy import copy

from pandas import DataFrame

from tsdm.datasets import InSilicoData
from tsdm.util import timefun


logger = logging.getLogger(__name__)


def test_caching():
    # NOTE: this test must be executed first!!!
    """Checks if dataset caching works (should be way faster!)"""

    ds = InSilicoData
    logger.info("Testing caching of dataset %s", ds.__name__)
    _, pre_cache_time = timefun(lambda: ds.dataset)()
    _, post_cache_time = timefun(lambda: ds.dataset)()

    logger.info("%f, %f", pre_cache_time, post_cache_time)

    assert 1000*post_cache_time <= pre_cache_time

    logger.info("%s passes caching test \N{HEAVY CHECK MARK}", ds.__name__)


def test_attributes():
    """Tests if all attributes are present"""

    ds = InSilicoData
    base_attrs = copy(set(dir(ds)))
    attrs = {'clean', 'dataset', 'dataset_file', 'dataset_path', 'download',
             'load', 'rawdata_path', 'url'}

    assert attrs <= base_attrs
    assert isinstance(ds.dataset, dict)

    ds = InSilicoData("new_url")
    instance_attrs = copy(set(dir(ds)))

    assert attrs <= instance_attrs
    assert isinstance(ds.dataset, dict)

    assert base_attrs == instance_attrs


if __name__ == "__main__":
    test_caching()
    test_attributes()
