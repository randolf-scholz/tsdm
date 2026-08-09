r"""Electricity Transformer Dataset (ETDataset).

**Source:** https://github.com/zhouhaoyi/ETDataset
"""

__all__ = ["ETT"]

from collections import defaultdict
from typing import Literal

from pandas import DataFrame, read_csv

from tsdm.datasets.base import DatasetBase

type ETT_Key = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


class ETT(DatasetBase[ETT_Key, DataFrame]):
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

    table_names = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
    rawdata_hashes = {
        "ETTh1.csv": "sha256:f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        "ETTh2.csv": "sha256:a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        "ETTm1.csv": "sha256:6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
        "ETTm2.csv": "sha256:db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
    }
    rawdata_schemas = defaultdict(
        lambda: {
            "date": "timestamp[ms][pyarrow]",
            "HUFL": "float32[pyarrow]",
            "HULL": "float32[pyarrow]",
            "MUFL": "float32[pyarrow]",
            "MULL": "float32[pyarrow]",
            "LUFL": "float32[pyarrow]",
            "LULL": "float32[pyarrow]",
            "OT": "float32[pyarrow]",
        }
    )
    dataset_hashes = {  # pyright: ignore[reportAssignmentType]
        "ETTh1": "sha256:b56abe3a5a0ac54428be73a37249d549440a7512fce182adcafba9ee43a03694",
        "ETTh2": "sha256:0607d0f59341e87f2ab0f520fb885ad6983aa5b17b058fc802ebd87c51f75387",
        "ETTm1": "sha256:62df6ea49e60b9e43e105b694e539e572ba1d06bda4df283faf53760d8cbd5c1",
        "ETTm2": "sha256:3c946e0fefc5c1a440e7842cdfeb7f6372a1b61b3da51519d0fb4ab8eb9debad",
    }
    table_shapes = {
        "ETTh1": (17420, 7),
        "ETTh2": (17420, 7),
        "ETTm1": (69680, 7),
        "ETTm2": (69680, 7),
    }

    def clean_table(self, key: ETT_Key) -> DataFrame:
        df = read_csv(
            self.rawdata_paths[f"{key}.csv"],
            index_col=0,
            dtype=self.rawdata_schemas[f"{key}.csv"],
            dtype_backend="pyarrow",
        )
        df.columns = df.columns.astype("string")
        return df
