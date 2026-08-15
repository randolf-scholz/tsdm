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
    ts = PolarsTS(
        timeseries=pl.DataFrame({"value": [10, 20, 30]}),
        timeindex=pl.Series("time", [1, 2, 3]),
    )

    actual = ts[key]

    assert isinstance(actual, PolarsTS)
    assert_frame_equal(actual.timeseries, pl.DataFrame({"value": expected_values}))
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
    collection = PolarsTSC(
        timeseries=pl.DataFrame({"value": [10, 20, 30, 40]}),
        timeindex=pl.Series("time", [1, 2, 1, 2]),
        metaindex=pl.Series("series", ["a", "a", "b", "b"]),
        static_covariates=pl.DataFrame({"offset": [100, 200]}),
    )

    actual = collection[key]

    assert isinstance(actual, expected_type)
    assert_frame_equal(actual.timeseries, pl.DataFrame({"value": expected_values}))
    assert actual.static_covariates is not None
    assert_frame_equal(
        actual.static_covariates,
        pl.DataFrame({"offset": [100] if expected_series == ["a"] else [200]}),
    )
    if isinstance(actual, PolarsTS):
        assert_series_equal(actual.timeindex, pl.Series("time", [1, 2]))
    else:
        assert_series_equal(
            actual.metaindex,
            pl.Series(
                "series",
                [series for series in expected_series for _ in range(2)],
            ),
        )

    with pytest.raises(KeyError):
        collection["missing"]
