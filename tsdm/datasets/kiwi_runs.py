r"""KIWI Run Data.

Extracted from iLab DataBase
"""

from __future__ import annotations

__all__ = [
    # Classes
    "KIWI_RUNS",
]

import logging
import os
import pickle
from functools import cache
from pathlib import Path
from typing import Final, Literal, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from tsdm.datasets.base import BaseDataset
from tsdm.util.strings import snake2camel

__logger__ = logging.getLogger(__name__)


def contains_no_information(series: Series) -> bool:
    return len(series.dropna().unique()) <= 1


def contains_nan_slice(
    series: Series, slices: list[Series], two_enough: bool = False
) -> bool:
    num_missing = 0
    for idx in slices:
        if pd.isna(series[idx]).all():
            num_missing += 1

    if (num_missing > 0 and not two_enough) or (
        num_missing >= len(slices) - 1 and two_enough
    ):
        __logger__.info(
            "%s: data missing in %s/%s slices!", series.name, num_missing, len(slices)
        )
        return True
    return False


def float_is_int(series: Series) -> bool:
    mask = pd.notna(series)
    return series[mask].apply(float.is_integer).all()


def get_integer_cols(table: DataFrame) -> set[str]:
    cols = set()
    for col in table:
        if np.issubdtype(table[col].dtype, np.integer):
            __logger__.info("Integer column                       : %s", col)
            cols.add(col)
        elif np.issubdtype(table[col].dtype, np.floating) and float_is_int(table[col]):
            __logger__.info("Integer column pretending to be float: %s", col)
            cols.add(col)
    return cols


def get_useless_cols(
    table: DataFrame, slices: Optional[list[Series]] = None, strict: bool = False
) -> set[str]:
    useless_cols = set()
    for col in table:
        s = table[col]
        if col in ("run_id", "experiment_id"):
            continue
        if contains_no_information(s):
            __logger__.info("No information in      %s", col)
            useless_cols.add(col)
        elif slices is not None and contains_nan_slice(
            s, slices, two_enough=(not strict)
        ):
            __logger__.info("Missing for some run   %s", col)
            useless_cols.add(col)
    return useless_cols


class KIWI_RUNS(BaseDataset):
    r"""KIWI RUN Data.

    The cleaned data will consist of 2 parts:

    - timeseries
    - metadata

    Rawdata Format:

    .. code-block:: python

        dict[int, # run_id
            dict[int, # experiment_id
                 dict[
                     'metadata',: DataFrame,                # static
                     'setpoints': DataFrame,                # static
                     'measurements_reactor',: DataFrame,    # TimeTensor
                     'measurements_array',: DataFrame,      # TimeTensor
                     'measurements_aggregated': DataFrame,  # TimeTensor
                 ]
            ]
        ]
    """

    url: str = (
        "https://owncloud.innocampus.tu-berlin.de/index.php/s/"
        "fRBSr82NxY7ratK/download/kiwi_experiments_and_run_355.pk"
    )
    keys: Final[list[str]] = [
        "metadata",
        "setpoints",
        "measurements_reactor",
        "measurements_array",
        "measurements_aggregated",
        "timeseries",
        "units",
    ]
    r"""Available keys."""
    KEYS = Literal[
        "metadata",
        "setpoints",
        "measurements_reactor",
        "measurements_array",
        "measurements_aggregated",
        "timeseries",
    ]
    r"""Type Hint for keys."""
    dataset: DataFrame
    r"""The main dataset. Alias for Timeseries."""
    timeseries: DataFrame
    r"""The main dataset."""

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def dataset(cls):
        r"""Store cached version of dataset."""
        # What is the best practice for metaclass methods that call each other?
        # https://stackoverflow.com/q/47615318/9318372
        if os.environ.get("GENERATING_DOCS", False):
            return "the dataset"
        return cls.load()  # pylint: disable=E1120

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def rawdata_file(cls) -> Path:
        r"""Path of the raw data file."""
        return cls.rawdata_path.joinpath("kiwi_experiments_and_run_355.pk")  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def dataset_file(cls) -> dict[str, Path]:
        r"""Path of the dataset file for the given key."""
        return {key: cls.dataset_path.joinpath(f"{key}.feather") for key in cls.keys}  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def metadata(cls) -> DataFrame:
        r"""Store cached version of dataset."""
        # What is the best practice for metaclass methods that call each other?
        # https://stackoverflow.com/q/47615318/9318372
        if os.environ.get("GENERATING_DOCS", False):
            return "the metadata"
        return cls.load("metadata")

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def timeseries(cls) -> DataFrame:
        """The TimeSeries."""
        return cls.dataset

    @classmethod
    @property
    @cache
    def units(cls) -> DataFrame:
        """The units metadata."""
        return cls.load("units")

    @classmethod
    def load(cls, key: KEYS = "timeseries") -> DataFrame:
        r"""Load the dataset from disk."""
        if not cls.dataset_file[key].exists():  # type: ignore[index]
            cls.clean()

        table = pd.read_feather(cls.dataset_file[key])  # type: ignore[index]

        if key == "units":
            return table.set_index("variable")

        # fix index dtype (groupby messes it up....)
        table = table.astype({"run_id": "int32", "experiment_id": "int32"})
        if "measurements" in key or key == "timeseries":
            table = table.set_index(["run_id", "experiment_id", "measurement_time"])
            table.columns.name = "variable"
        else:
            table = table.set_index(["run_id", "experiment_id"])
        return table

    @classmethod
    def clean(cls):
        r"""Clean an already downloaded raw dataset and stores it in feather format."""
        dataset = cls.__name__
        __logger__.info("Cleaning dataset '%s'", dataset)

        with open(cls.rawdata_file, "rb") as file:  # type: ignore[call-overload]
            data = pickle.load(file)

        DATA = [
            (data[run][exp] | {"run_id": run, "experiment_id": exp})
            for run in data
            for exp in data[run]
        ]
        DF = DataFrame(DATA).set_index(["run_id", "experiment_id"])

        tables = {}

        for key in cls.keys:
            if key in ("units", "timeseries"):
                cls._clean(key)
            elif key == "metadata":
                tables[key] = pd.concat(iter(DF[key])).reset_index(drop=True)
                cls._clean(key, tables[key])
            else:
                tables[key] = (
                    pd.concat(iter(DF[key]), keys=DF[key].index)
                    .reset_index(level=2, drop=True)
                    .reset_index()
                )
                cls._clean(key, tables[key])

        __logger__.info("Finished cleaning dataset '%s'", dataset)

    @classmethod
    def _clean(cls, key: str, table: Optional[DataFrame] = None):
        r"""Create the DataFrames.

        Parameters
        ----------
        table: DataFrame
        """
        cleaner = {
            "units": cls._clean_units,
            "timeseries": cls._clean_timeseries,
            "metadata": cls._clean_metadata,
            "setpoints": cls._clean_setpoints,
            "measurements_reactor": cls._clean_measurements_reactor,
            "measurements_array": cls._clean_measurements_array,
            "measurements_aggregated": cls._clean_measurements_aggregated,
        }[key]

        cleaner() if table is None else cleaner(table)

        __logger__.info(
            "Finished cleaning table '%s' of dataset '%s'", key, cls.__name__
        )

    @classmethod
    def _clean_metadata(cls, table):
        runs = table["run_id"].dropna().unique()
        run_masks = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks) | {
            "folder_id_y",
            "ph_Base_conc",
            "ph_Ki",
            "ph_Kp",
            "ph_Tolerance",
            "pms_id",
            "description",
        }
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns = {
            "Feed_concentration_glc": "float32",
            "OD_Dilution": "float32",
            "bioreactor_id": "UInt32",
            "color": "string",
            "container_number": "UInt32",
            "end_time": "datetime64[ns]",
            "experiment_id": "UInt32",
            "organism_id": "UInt32",
            "pH_correction_factor": "float32",
            "profile_id": "UInt32",
            "profile_name": "string",
            "run_id": "UInt32",
            "run_name": "string",
            "start_time": "datetime64[ns]",
        }

        categorical_columns = {
            "Feed_concentration_glc": "Int16",
            "OD_Dilution": "Float32",
            "color": "category",
            "pH_correction_factor": "Float32",
            "profile_name": "category",
            "run_name": "category",
        }

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table = table[selected_columns]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        table.columns.name = "variable"
        path = cls.dataset_file["metadata"]  # type: ignore[index]
        table.to_feather(path)

    @classmethod
    def _clean_setpoints(cls, table):
        runs = table["run_id"].dropna().unique()
        run_masks = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns = {
            "experiment_id": "UInt32",
            "run_id": "UInt32",
            "cultivation_age": "UInt32",
            "setpoint_id": "UInt32",
            "unit": "string",
            # "Puls_AceticAcid": "Float32",
            "Puls_Glucose": "Float32",
            # "Puls_Medium": "Float32",
            "StirringSpeed": "UInt16",
            # "pH": "Float32",
            "Feed_glc_cum_setpoints": "UInt16",
            "Flow_Air": "UInt8",
            "InducerConcentration": "Float32",
            # "Flow_Nitrogen": "Float32",
            # "Flow_O2": "Float32",
            # "Feed_dextrine_cum_setpoints": "Float32",
        }

        categorical_columns = {
            "unit": "category",
        }

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        path = cls.dataset_file["setpoints"]  # type: ignore[index]
        table.to_feather(path)

    @classmethod
    def _clean_measurements_reactor(cls, table: DataFrame):
        runs = table["run_id"].dropna().unique()
        run_masks = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns: dict[str, str] = {
            "Acetate": "Float32",
            "Base": "Int16",
            "Cumulated_feed_volume_glucose": "Int16",
            "Cumulated_feed_volume_medium": "Float32",
            "DOT": "Float32",
            "Fluo_GFP": "Float32",
            "Glucose": "Float32",
            "InducerConcentration": "Float32",
            "OD600": "Float32",
            "Probe_Volume": "Int16",
            "Volume": "Float32",
            "experiment_id": "UInt32",
            "measurement_id": "UInt32",
            "measurement_time": "datetime64[ns]",
            "pH": "Float32",
            "run_id": "UInt32",
            "unit": "string",
        }

        categorical_columns: dict[str, str] = {
            "unit": "category",
        }

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        path = cls.dataset_file["measurements_reactor"]  # type: ignore[index]
        table.to_feather(path)

    @classmethod
    def _clean_measurements_array(cls, table: DataFrame):
        runs = table["run_id"].dropna().unique()
        run_masks = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns: dict[str, str] = {
            "run_id": "UInt32",
            "experiment_id": "UInt32",
            "measurement_time": "datetime64[ns]",
            "measurement_id": "UInt32",
            "unit": "string",
            "Flow_Air": "Float32",
            # "Flow_Nitrogen"      :         "float64",
            # "Flow_O2"            :         "float64",
            "StirringSpeed": "Int16",
            "Temperature": "Float32",
        }

        categorical_columns: dict[str, str] = {
            "unit": "category",
        }

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        path = cls.dataset_file["measurements_array"]  # type: ignore[index]
        table.to_feather(path)

    @classmethod
    def _clean_measurements_aggregated(cls, table: DataFrame):
        runs = table["run_id"].dropna().unique()
        run_masks = [table["run_id"] == run for run in runs]
        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns: dict[str, str] = {
            "run_id": "UInt32",
            "experiment_id": "UInt32",
            "measurement_time": "datetime64[ns]",
            "unit": "string",
            "Flow_Air": "Float32",
            # "Flow_Nitrogen"                 :          "Float32",
            # "Flow_O2"                       :          "Int32",
            "StirringSpeed": "Int16",
            "Temperature": "Float32",
            "Acetate": "Float32",
            # "Acid"                          :          "Float32",
            "Base": "Int16",
            "Cumulated_feed_volume_glucose": "Int16",
            "Cumulated_feed_volume_medium": "Float32",
            "DOT": "Float32",
            # "Fluo_CFP"                      :          "Float32",
            # "Fluo_RFP"                      :          "Float32",
            # "Fluo_YFP"                      :          "Float32",
            "Glucose": "Float32",
            "OD600": "Float32",
            "Probe_Volume": "Int16",
            "pH": "Float32",
            "Fluo_GFP": "Float32",
            "InducerConcentration": "Float32",
            # "remark"                        :           "string",
            "Volume": "Float32",
        }
        categorical_columns: dict[str, str] = {"unit": "category"}

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        path = cls.dataset_file["measurements_aggregated"]  # type: ignore[index]
        table.to_feather(path)

    @classmethod
    def _clean_timeseries(cls):
        md = cls.load("metadata")
        ts = cls.load("measurements_aggregated")
        # drop rows with only <NA> values
        ts = ts.dropna(how="all")
        # generate timeseries frame
        ts = ts.drop(columns="unit")
        # gather
        ts = ts.groupby(["run_id", "experiment_id", "measurement_time"]).mean()
        # drop rows with only <NA> values
        ts = ts.dropna(how="all")
        # convert all value columns to float
        ts = ts.astype("Float32")
        # sort according to measurement_time
        ts = ts.sort_values(["run_id", "experiment_id", "measurement_time"])
        # drop all time stamps outside of start_time and end_time in the metadata.
        merged = ts[[]].join(md[["start_time", "end_time"]])
        time = merged.index.get_level_values("measurement_time")
        cond = (merged["start_time"] <= time) & (time <= merged["end_time"])
        ts = ts[cond]

        # replace wrong pH values
        ts["pH"] = ts["pH"].replace(0.0, pd.NA)
        # mask out-of bounds values
        ts["DOT"][ts["DOT"] < 0] = pd.NA
        ts["DOT"][ts["DOT"] > 100] = 100.0
        ts["Glucose"][ts["Glucose"] < 0] = pd.NA
        ts["Acetate"][ts["Acetate"] < 0] = pd.NA
        ts["OD600"][ts["OD600"] < 0] = pd.NA
        # drop rows with only <NA> values
        ts = ts.dropna(how="all")

        # Forward Fill piece-wise constant control variables
        ffill_consts = [
            "Cumulated_feed_volume_glucose",
            "Cumulated_feed_volume_medium",
            "InducerConcentration",
            "StirringSpeed",
            "Flow_Air",
            "Probe_Volume",
        ]
        for idx, slc in ts.groupby(["run_id", "experiment_id"]):
            slc = slc[ffill_consts].fillna(method="ffill")
            # forward fill remaining NA with zeros.
            ts.loc[idx, ffill_consts] = slc[ffill_consts].fillna(0)

        # check if metadata-index matches with times-series index
        tsidx = ts.reset_index(level="measurement_time").index
        pd.testing.assert_index_equal(md.index, tsidx.unique())

        # reset index
        ts = ts.reset_index()
        ts = ts.astype({"run_id": "int32", "experiment_id": "int32"})
        # ts = ts.rename(columns={col: snake2camel(col) for col in ts})
        ts.columns.name = "variable"
        path = cls.dataset_file["timeseries"]  # type: ignore[index]
        ts.to_feather(path)

    @classmethod
    def _clean_units(cls):
        ts = cls.load("measurements_aggregated")

        _units = ts["unit"]
        data = ts.drop(columns="unit")
        data = data.astype("float32")

        units = Series(dtype=pd.StringDtype(), name="unit")

        for col in data:
            if col == "runtime": continue
            mask = pd.notna(data[col])
            unit = _units[mask].unique().to_list()
            assert len(unit) <= 1, f"{col}, {unit} {len(unit)}"
            units[col] = unit[0]

        units["DOT"] = "%"
        units["OD600"] = "%"
        units["pH"] = "-log[H⁺]"
        units["Acetate"] = "g/L"
        units = units.fillna(pd.NA).astype("string").astype("category")
        units.index.name = "variable"
        units = units.reset_index()
        path = cls.dataset_file["units"]  # type: ignore[index]
        units.to_feather(path)
