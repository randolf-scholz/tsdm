r"""Testing of In Silico dataset, as a token for the whole BaseDataset architecture."""

import logging

import polars as pl

from tsdm.datasets import BaseDataset, Dataset, InSilico
from tsdm.utils import timer

__logger__ = logging.getLogger(__name__)


def test_insilico_preprocessing() -> None:
    InSilico.reset_dataset_files(force=True)
    ds = InSilico()

    assert ds.timeseries_metadata.height == ds.timeseries.width

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]


def test_caching() -> None:
    r"""Test the caching of the dataset."""
    LOGGER = __logger__.getChild(InSilico.__name__)
    LOGGER.info("Testing caching.")

    ds = InSilico(initialize=False)

    with timer() as t:
        _ = ds.timeseries

    pre_cache_time = t.elapsed_time

    with timer() as t:
        _ = ds.timeseries

    post_cache_time = t.elapsed_time

    LOGGER.info("%f, %f", pre_cache_time, post_cache_time)

    assert 100 * post_cache_time <= pre_cache_time, (
        f"{post_cache_time=}, {pre_cache_time=}"
    )

    LOGGER.info("%s passes caching test ✔.")


def test_dataset_protocol() -> None:
    r"""Test the attributes of the dataset."""
    LOGGER = __logger__.getChild(InSilico.__name__)
    LOGGER.info("Testing attributes.")
    ds = InSilico()
    assert isinstance(ds, Dataset)
    assert isinstance(ds, BaseDataset)
