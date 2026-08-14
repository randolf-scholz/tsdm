r"""Preprocessed subset of the USHCN climate dataset used by De Brouwer et al."""

__all__ = ["RAWDATA_SCHEMA", "TIMESERIES_SCHEMA", "USHCN_DeBrouwer2019"]

from typing import Literal

import polars as pl

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import validate_schema

RAWDATA_SCHEMA = {
    "ID": pl.Float32,
    "Time": pl.Float32,
    "Value_0": pl.Float32,
    "Value_1": pl.Float32,
    "Value_2": pl.Float32,
    "Value_3": pl.Float32,
    "Value_4": pl.Float32,
    "Mask_0": pl.Float32,
    "Mask_1": pl.Float32,
    "Mask_2": pl.Float32,
    "Mask_3": pl.Float32,
    "Mask_4": pl.Float32,
}
TIMESERIES_SCHEMA = {
    "ID"     : pl.Int16,
    "Time"   : pl.Float32,
    "Value_0": pl.Float32,
    "Value_1": pl.Float32,
    "Value_2": pl.Float32,
    "Value_3": pl.Float32,
    "Value_4": pl.Float32,
}  # fmt: skip


class USHCN_DeBrouwer2019(DatasetBase[Literal["timeseries"], pl.DataFrame]):
    r"""Preprocessed subset of the USHCN climate dataset used by De Brouwer et al.

    References:
        - | `GRU-ODE-Bayes: Continuous Modeling of Sporadically﹣Observed Time Series
            <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>`_
          | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves
          | `Advances in Neural Information Processing Systems 2019<https://proceedings.neurips.cc/paper/2019>`_
    """

    SOURCE_URL = r"https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/master/gru_ode_bayes/datasets/Climate/"
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = "https://github.com/edebrouwer/gru_ode_bayes"
    r"""HTTP address containing additional information about the dataset."""

    table_names = ["timeseries"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["small_chunked_sporadic.csv"]
    rawdata_hashes = {
        "small_chunked_sporadic.csv": "sha256:671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf",
    }
    rawdata_schemas = {"small_chunked_sporadic.csv": RAWDATA_SCHEMA}
    table_schemas = {"timeseries": TIMESERIES_SCHEMA}
    table_shapes = {"timeseries": (350_665, 7)}

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Clean the raw USHCN subset into a masked Polars timeseries table."""
        fname = self.rawdata_files[0]
        rawdata_schema = self.rawdata_schemas[fname]
        target_schema = self.table_schemas["timeseries"]
        rawdata_path = self.rawdata_paths[fname]
        validate_schema(rawdata_path, rawdata_schema)
        table = pl.read_csv(rawdata_path, schema=rawdata_schema)

        return table.select(
            pl.col("ID").cast(target_schema["ID"]),
            pl.col("Time"),
            *(
                (
                    pl.when(pl.col(f"Mask_{index}").eq(1))
                    .then(pl.col(f"Value_{index}"))
                    .otherwise(None)
                )
                for index in range(5)
            ),
        ).sort("ID", "Time")

    def load_table(self, key: Literal["timeseries"], /) -> pl.DataFrame:
        r"""Load the cleaned USHCN subset as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])
