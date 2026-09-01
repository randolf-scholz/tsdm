from collections.abc import Mapping

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from tsdm.timeseries.pandas import (
    TimeSeries,
    TimeSeriesCollection,
    beijing_air_quality,
    make_sample_factory,
)


def test_beijing_air_quality() -> None:
    ds = beijing_air_quality()
    assert isinstance(ds, TimeSeriesCollection)
    assert isinstance(ds.timeseries, pd.DataFrame)
    assert isinstance(ds.timeseries.index, pd.MultiIndex)
    assert ds.timeseries.index.names == ["station", "time"]
    metadata = ds.timeseries_metadata
    assert metadata is not None
    assert isinstance(metadata, pd.DataFrame)
    assert metadata.index.name == "variable"


@pytest.mark.parametrize(
    ("key", "expected_index"),
    [
        (2, [2]),
        ([1, 3], [1, 3]),
        ([False, True, False], [2]),
        (slice(1, 2), [1, 2]),
    ],
)
def test_timeseries_getitem(
    key: int | slice | list[int] | list[bool], expected_index: list[int]
) -> None:
    timeseries = pd.DataFrame(
        {"value": [10, 20, 30]},
        index=pd.Index([1, 2, 3], name="time"),
    )
    ts = TimeSeries(timeseries=timeseries)

    actual = ts[key]

    assert isinstance(actual, TimeSeries)
    assert_series_equal(
        actual.timeindex,
        pd.Series(
            expected_index,
            index=pd.Index(expected_index, name="time"),
            name="time",
        ),
    )
    assert_frame_equal(actual.timeseries, timeseries.loc[expected_index])

    with pytest.raises(KeyError):
        ts[4]


@pytest.mark.parametrize(
    ("key", "expected_type", "expected_series"),
    [
        ("a", TimeSeries, ["a"]),
        (["b"], TimeSeriesCollection, ["b"]),
        ([False, True], TimeSeriesCollection, ["b"]),
        (slice("a", "a"), TimeSeriesCollection, ["a"]),
    ],
)
def test_timeseries_collection_getitem(
    key: str | slice | list[str] | list[bool],
    expected_type: type[TimeSeries | TimeSeriesCollection],
    expected_series: list[str],
) -> None:
    index = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)],
        names=["series", "time"],
    )
    collection = TimeSeriesCollection(
        timeseries=pd.DataFrame({"value": [10, 20, 30, 40]}, index=index),
        static_covariates=pd.DataFrame(
            {"offset": [100, 200]}, index=pd.Index(["a", "b"], name="series")
        ),
    )
    assert isinstance(collection, Mapping)
    assert list(collection.keys()) == ["a", "b"]
    assert all(isinstance(value, TimeSeries) for value in collection.values())

    actual = collection[key]

    assert isinstance(actual, expected_type)
    assert actual.timeseries["value"].tolist() == (
        [10, 20] if expected_series == ["a"] else [30, 40]
    )
    assert actual.static_covariates is not None
    assert actual.static_covariates.index.tolist() == expected_series
    if isinstance(actual, TimeSeries):
        assert_series_equal(
            actual.timeindex,
            pd.Series([1, 2], index=pd.Index([1, 2], name="time"), name="time"),
        )
    else:
        assert_series_equal(
            actual.timeindex,
            pd.Series(
                [1, 2] * len(expected_series),
                index=pd.Index(
                    [series for series in expected_series for _ in range(2)],
                    name="series",
                ),
                name="time",
            ),
        )
        assert actual.metaindex.tolist() == expected_series

    with pytest.raises(KeyError):
        collection["missing"]


def test_timeseries_collection_timeindex_uses_row_metaindex() -> None:
    index = pd.MultiIndex.from_tuples(
        [(1, 1, 1), (1, 1, 2), (1, 2, 1)],
        names=["batch", "series", "time"],
    )
    collection = TimeSeriesCollection(
        timeseries=pd.DataFrame({"value": [10, 20, 30]}, index=index)
    )
    expected_timeindex = pd.Series(
        [1, 2, 1],
        index=pd.MultiIndex.from_tuples(
            [(1, 1), (1, 1), (1, 2)], names=["batch", "series"]
        ),
        name="time",
    )

    assert_series_equal(collection.timeindex, expected_timeindex)
    assert_index_equal(collection.metaindex, expected_timeindex.index.unique())


def test_forecasting_sample_static_covariates_from_timeseries() -> None:
    static_covariates = pd.DataFrame({"offset": [100]})
    timeseries = TimeSeries(
        timeseries=pd.DataFrame({"value": [10, 20, 30]}),
        static_covariates=static_covariates,
    )
    make_sample = make_sample_factory(
        timeseries,
        targets=["value"],
        observables=["value"],
        covariates=[],
    )

    sample = make_sample[[slice(0, 1), slice(2, 2)]]

    assert sample.static_covariates is static_covariates


def test_forecasting_sample_static_covariates_from_collection() -> None:
    index = pd.MultiIndex.from_product(
        [["a", "b"], [0, 1, 2]], names=["series", "time"]
    )
    static_covariates = pd.DataFrame(
        {"offset": [100, 200]}, index=pd.Index(["a", "b"], name="series")
    )
    collection = TimeSeriesCollection(
        timeseries=pd.DataFrame({"value": range(6)}, index=index),
        static_covariates=static_covariates,
    )
    make_sample = make_sample_factory(
        collection,
        targets=["value"],
        observables=["value"],
        covariates=[],
    )

    sample = make_sample["a", [slice(0, 1), slice(2, 2)]]

    assert isinstance(sample.static_covariates, pd.Series)
    assert_series_equal(sample.static_covariates, static_covariates.loc["a"])


def test_make_sample_factory_validates_selected_columns() -> None:
    r"""Column validation is performed while constructing the sample factory."""
    timeseries = TimeSeries(timeseries=pd.DataFrame({"value": [10, 20, 30]}))

    with pytest.raises(ValueError, match="Covariates and observables"):
        make_sample_factory(
            timeseries,
            targets=["value"],
            observables=["value"],
            covariates=["value"],
        )
