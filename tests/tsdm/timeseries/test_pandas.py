import pandas as pd

from tsdm.timeseries.pandas import PandasTSC, beijing_air_quality


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
