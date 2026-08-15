import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tsdm.timeseries.pandas import PandasTS, PandasTSC, beijing_air_quality


def test_beijing_air_quality() -> None:
    ds = beijing_air_quality()
    assert isinstance(ds, PandasTSC)
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
    ts = PandasTS(timeseries=timeseries)

    actual = ts[key]

    assert isinstance(actual, PandasTS)
    assert actual.timeindex.tolist() == expected_index
    assert_frame_equal(actual.timeseries, timeseries.loc[expected_index])

    with pytest.raises(KeyError):
        ts[4]


@pytest.mark.parametrize(
    ("key", "expected_type", "expected_series"),
    [
        ("a", PandasTS, ["a"]),
        (["b"], PandasTSC, ["b"]),
        ([False, True], PandasTSC, ["b"]),
        (slice("a", "a"), PandasTSC, ["a"]),
    ],
)
def test_timeseries_collection_getitem(
    key: str | slice | list[str] | list[bool],
    expected_type: type[PandasTS | PandasTSC],
    expected_series: list[str],
) -> None:
    index = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)],
        names=["series", "time"],
    )
    collection = PandasTSC(
        timeseries=pd.DataFrame({"value": [10, 20, 30, 40]}, index=index),
        static_covariates=pd.DataFrame(
            {"offset": [100, 200]}, index=pd.Index(["a", "b"], name="series")
        ),
    )

    actual = collection[key]

    assert isinstance(actual, expected_type)
    assert actual.timeseries["value"].tolist() == (
        [10, 20] if expected_series == ["a"] else [30, 40]
    )
    assert actual.static_covariates is not None
    assert actual.static_covariates.index.tolist() == expected_series
    if isinstance(actual, PandasTS):
        assert actual.timeindex.tolist() == [1, 2]
    else:
        assert actual.metaindex.tolist() == expected_series

    with pytest.raises(KeyError):
        collection["missing"]
