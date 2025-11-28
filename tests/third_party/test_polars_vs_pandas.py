import datetime as dt

import pandas as pd
import polars as pl


def test_timestamp_to_float() -> None:
    timestamps: list[dt.datetime] = [
        dt.datetime(2020, 1, 13, 11, 0),
        dt.datetime(2021, 6, 15, 12, 30),
        dt.datetime(2022, 12, 31, 23, 59),
    ]
    values: list[float] = [
        0.0,
        1.4548611111111112,
        2.997267361111111,
    ]

    reference_time = dt.datetime(2020, 1, 1, 0, 0)
    time_unit = dt.timedelta(seconds=1)

    pd_series = pd.Series(timestamps)
    pl_series = pl.Series("timestamps", timestamps)

    pd_encoded = (pd_series - reference_time) / time_unit
    pd_decoded = (pd_encoded * time_unit) + reference_time
    assert (
        all(pd_decoded == pd_series) and pd_decoded.dtype.kind == pd_series.dtype.kind
    )

    pl_encoded = (pl_series - reference_time) / time_unit
    pl_decoded = (time_unit * pl_encoded) + reference_time
    assert all(pl_decoded == pl_series) and type(pl_decoded.dtype) == type(
        pl_series.dtype
    )
