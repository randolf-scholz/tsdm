from collections.abc import Mapping

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from tsdm.timeseries.polars import PolarsTS, PolarsTSC


@pytest.mark.parametrize(
    ("key", "expected_timeindex", "expected_values"),
    [
        (2, [2], [20]),
        ([1, 3], [1, 3], [10, 30]),
        ([False, True, False], [2], [20]),
        (slice(1, 2), [1, 2], [10, 20]),
    ],
)
def test_timeseries_getitem(
    key: int | slice | list[int] | list[bool],
    expected_timeindex: list[int],
    expected_values: list[int],
) -> None:
    raw_data = {"time": [1, 2, 3], "value": [10, 20, 30]}
    ts = PolarsTS(
        timeseries=pl.DataFrame(raw_data),
        time_column="time",
    )

    actual = ts[key]

    assert isinstance(actual, PolarsTS)
    assert_frame_equal(
        actual.timeseries,
        pl.DataFrame({"time": expected_timeindex, "value": expected_values}),
    )
    assert_series_equal(actual.timeindex, pl.Series("time", expected_timeindex))

    with pytest.raises(KeyError):
        ts[4]


@pytest.mark.parametrize(
    ("key", "expected_type", "expected_series", "expected_values"),
    [
        ("a", PolarsTS, ["a"], [10, 20]),
        (["b"], PolarsTSC, ["b"], [30, 40]),
        ([False, True], PolarsTSC, ["b"], [30, 40]),
        (slice("a", "a"), PolarsTSC, ["a"], [10, 20]),
    ],
)
def test_timeseries_collection_getitem(
    key: str | slice | list[str] | list[bool],
    expected_type: type[PolarsTS | PolarsTSC],
    expected_series: list[str],
    expected_values: list[int],
) -> None:
    raw_data = {
        "series": ["a", "a", "b", "b"],
        "time": [1, 2, 1, 2],
        "value": [10, 20, 30, 40],
    }
    collection = PolarsTSC(
        timeseries=pl.DataFrame(raw_data),
        time_column="time",
        meta_columns=["series"],
        static_covariates=pl.DataFrame({"series": ["a", "b"], "offset": [100, 200]}),
    )
    assert isinstance(collection, Mapping)
    assert list(collection.keys()) == ["a", "b"]
    assert all(isinstance(value, PolarsTS) for value in collection.values())
    assert_frame_equal(collection.metaindex, pl.DataFrame({"series": ["a", "b"]}))

    actual = collection[key]

    assert isinstance(actual, expected_type)
    assert_frame_equal(
        actual.timeseries,
        pl.DataFrame(
            {
                "series": [series for series in expected_series for _ in range(2)],
                "time": [1, 2] * len(expected_series),
                "value": expected_values,
            }
        ),
    )
    assert actual.static_covariates is not None
    assert_frame_equal(
        actual.static_covariates,
        pl.DataFrame(
            {
                "series": expected_series,
                "offset": [100 if series == "a" else 200 for series in expected_series],
            }
        ),
    )
    assert_series_equal(
        actual.timeindex,
        pl.Series("time", [1, 2] * len(expected_series)),
    )
    if isinstance(actual, PolarsTSC):
        assert_frame_equal(
            actual.metaindex,
            pl.DataFrame({"series": expected_series}),
        )

    with pytest.raises(KeyError):
        collection["missing"]


def test_timeseries_collection_getitem_with_multicolumn_metaindex() -> None:
    collection = PolarsTSC(
        timeseries=pl.DataFrame(
            {
                "batch": [1, 1, 1, 1, 2, 2],
                "series": [1, 1, 2, 2, 1, 1],
                "time": [1, 2, 1, 2, 1, 2],
                "value": [10, 20, 30, 40, 50, 60],
            }
        ),
        time_column="time",
        meta_columns=["batch", "series"],
        static_covariates=pl.DataFrame(
            {
                "batch": [1, 1, 2],
                "series": [1, 2, 1],
                "offset": [100, 200, 300],
            }
        ),
    )

    assert list(collection) == [(1, 1), (1, 2), (2, 1)]
    assert_frame_equal(
        collection.metaindex,
        pl.DataFrame({"batch": [1, 1, 2], "series": [1, 2, 1]}),
    )
    assert (1, 2) in collection
    assert (2, 2) not in collection

    actual = collection[(1, 2)]

    assert isinstance(actual, PolarsTS)
    assert_frame_equal(
        actual.timeseries,
        pl.DataFrame(
            {
                "batch": [1, 1],
                "series": [2, 2],
                "time": [1, 2],
                "value": [30, 40],
            }
        ),
    )
    assert actual.static_covariates is not None
    assert_frame_equal(
        actual.static_covariates,
        pl.DataFrame({"batch": [1], "series": [2], "offset": [200]}),
    )
    assert_series_equal(actual.timeindex, pl.Series("time", [1, 2]))

    subset = collection[[(1, 1), (2, 1)]]

    assert isinstance(subset, PolarsTSC)
    assert_frame_equal(
        subset.metaindex,
        pl.DataFrame({"batch": [1, 2], "series": [1, 1]}),
    )
    assert subset.static_covariates is not None
    assert_frame_equal(
        subset.static_covariates,
        pl.DataFrame({"batch": [1, 2], "series": [1, 1], "offset": [100, 300]}),
    )

    with pytest.raises(KeyError):
        collection[(2, 2)]
    with pytest.raises(ValueError, match="Cannot slice a multi-column metaindex"):
        collection[:]


def test_timeseries_collection_requires_static_covariate_meta_columns() -> None:
    with pytest.raises(ValueError, match="contain all meta_columns"):
        PolarsTSC(
            timeseries=pl.DataFrame(
                {
                    "series": ["a", "a", "b", "b"],
                    "time": [1, 2, 1, 2],
                    "value": [10, 20, 30, 40],
                }
            ),
            time_column="time",
            meta_columns=["series"],
            static_covariates=pl.DataFrame({"offset": [100, 200]}),
        )
