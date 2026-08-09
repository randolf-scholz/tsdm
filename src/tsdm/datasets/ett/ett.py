r"""Electricity Transformer Dataset (ETDataset).

**Source:** https://github.com/zhouhaoyi/ETDataset
"""

__all__ = ["ETT"]

from typing import Literal

import polars as pl

from tsdm.datasets.base import DatasetBase

type ETT_Key = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

_ETT_KEYS: tuple[ETT_Key, ...] = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
_ETT_SCHEMA = {
    "date": pl.Datetime(time_unit="us"),
    "HUFL": pl.Float32,
    "HULL": pl.Float32,
    "MUFL": pl.Float32,
    "MULL": pl.Float32,
    "LUFL": pl.Float32,
    "LULL": pl.Float32,
    "OT": pl.Float32,
}


class ETT(DatasetBase[ETT_Key, pl.DataFrame]):
    r"""ETT dataset.

    This dataset contains 4 variants: ETTh1, ETTh2, ETTm1, ETTm2, which contain time series data
    from two electrical transformers (1 and 2) with hourly (h) and minute (m) resolution.

    +-------+--------------------------+
    | Field | Description              |
    +=======+==========================+
    | date  | The recorded date        |
    +-------+--------------------------+
    | HUFL  | High UseFul Load         |
    +-------+--------------------------+
    | HULL  | High UseLess Load        |
    +-------+--------------------------+
    | MUFL  | Middle UseFul Load       |
    +-------+--------------------------+
    | MULL  | Middle UseLess Load      |
    +-------+--------------------------+
    | LUFL  | Low UseFul Load          |
    +-------+--------------------------+
    | LULL  | Low UseLess Load         |
    +-------+--------------------------+
    | OT    | Oil Temperature (target) |
    +-------+--------------------------+
    """

    SOURCE_URL = r"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = r"https://github.com/zhouhaoyi/ETDataset"
    r"""HTTP address containing additional information about the dataset."""

    table_names = _ETT_KEYS
    rawdata_files = [f"{key}.csv" for key in _ETT_KEYS]
    rawdata_hashes = {
        "ETTh1.csv": "sha256:f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        "ETTh2.csv": "sha256:a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        "ETTm1.csv": "sha256:6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
        "ETTm2.csv": "sha256:db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
    }
    rawdata_schemas = dict.fromkeys((f"{key}.csv" for key in _ETT_KEYS), _ETT_SCHEMA)
    table_schemas = dict.fromkeys(_ETT_KEYS, _ETT_SCHEMA)
    table_shapes = {
        "ETTh1": (17420, 8),
        "ETTh2": (17420, 8),
        "ETTm1": (69680, 8),
        "ETTm2": (69680, 8),
    }

    def clean_table(self, key: ETT_Key, /) -> pl.DataFrame:
        r"""Load an ETT CSV file as a Polars DataFrame."""
        return pl.read_csv(
            self.rawdata_paths[f"{key}.csv"],
            schema=self.rawdata_schemas[f"{key}.csv"],
        )

    def load_table(self, key: ETT_Key, /) -> pl.DataFrame:
        r"""Load a cleaned ETT table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])
