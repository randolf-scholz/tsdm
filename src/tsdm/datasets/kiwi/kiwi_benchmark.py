r"""The KIWI Benchmark Dataset."""

__all__ = [
    "STATIC_COVARIATES_METADATA_SCHEMA",
    "STATIC_COVARIATES_SCHEMA",
    "TIMESERIES_METADATA_SCHEMA",
    "TIMESERIES_SCHEMA",
    "KiwiBenchmark",
]

import shutil
from importlib import resources
from typing import Literal
from zipfile import ZipFile

import polars as pl

from tsdm.datasets.base import DatasetBase

TIMESERIES_SCHEMA = {
    "run_id"                        : pl.UInt64,
    "experiment_id"                 : pl.UInt64,
    "elapsed_time"                  : pl.Duration(time_unit="ms"),
    "Acetate"                       : pl.Float32,
    "Base"                          : pl.Float32,
    "Cumulated_feed_volume_glucose" : pl.Float32,
    "Cumulated_feed_volume_medium"  : pl.Float32,
    "DOT"                           : pl.Float32,
    "Fluo_GFP"                      : pl.Float32,
    "Glucose"                       : pl.Float32,
    "OD600"                         : pl.Float32,
    "Probe_Volume"                  : pl.Float32,
    "pH"                            : pl.Float32,
    "Flow_Air"                      : pl.Float32,
    "StirringSpeed"                 : pl.Float32,
    "Temperature"                   : pl.Float32,
    "InducerConcentration"          : pl.Float32,
}  # fmt: skip
TIMESERIES_METADATA_SCHEMA = {
    "dtype"          : pl.String,
    "type"           : pl.String,
    "kind"           : pl.String,
    "unit"           : pl.String,
    "lower_bound"    : pl.Float32,
    "upper_bound"    : pl.Float32,
    "lower_included" : pl.Boolean,
    "upper_included" : pl.Boolean,
    "description"    : pl.String,
    "name"           : pl.String,
}  # fmt: skip
STATIC_COVARIATES_SCHEMA = {
    "run_id"                 : pl.UInt64,
    "experiment_id"          : pl.UInt64,
    "start_time"             : pl.Datetime(time_unit="ms"),
    "end_time"               : pl.Datetime(time_unit="ms"),
    "bioreactor_type_name"   : pl.String,
    "description"            : pl.String,
    "color"                  : pl.String,
    "medium_name"            : pl.String,
    "organism_name"          : pl.String,
    "plasmid_name"           : pl.String,
    "profile_name"           : pl.String,
    "Acetate_Dilution"       : pl.Float32,
    "Feed_concentration_glc" : pl.Float32,
    "Glucose_Dilution"       : pl.Float32,
    "InducerConcentration"   : pl.Float32,
    "OD_Dilution"            : pl.Float32,
    "Stir_Max_Restarts"      : pl.Float32,
    "capacity_per_container" : pl.Float32,
    "pH_correction_factor"   : pl.Float32,
    "ph_Acid_conc"           : pl.Float32,
    "ph_Base_conc"           : pl.Float32,
    "ph_Ki"                  : pl.Float32,
    "ph_Kp"                  : pl.Float32,
    "ph_Tolerance"           : pl.Float32,
}  # fmt: skip
STATIC_COVARIATES_METADATA_SCHEMA = {
    "dtype"          : pl.String,
    "kind"           : pl.String,
    "unit"           : pl.String,
    "lower_bound"    : pl.Float32,
    "upper_bound"    : pl.Float32,
    "lower_included" : pl.Boolean,
    "upper_included" : pl.Boolean,
    "description"    : pl.String,
    "name"           : pl.String,
}  # fmt: skip


type Key = Literal[
    "timeseries",
    "timeseries_metadata",
    "static_covariates",
    "static_covariates_metadata",
]


class KiwiBenchmark(DatasetBase[Key, pl.DataFrame]):
    r"""KIWI Benchmark Dataset."""

    DEFAULT_VERSION = "1.0"
    version: str  # pyright: ignore[reportIncompatibleMethodOverride]

    INFO_URL = r"https://www.tu.berlin/bioprocess/einrichtungen-associates/arbeitsgruppen/kiwi-biolab"
    HOME_URL = r"https://www.tu.berlin/bioprocess/einrichtungen-associates/arbeitsgruppen/kiwi-biolab"

    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "timeseries_metadata",
        "static_covariates",
        "static_covariates_metadata",
    ]
    rawdata_files = ["kiwi-benchmark.zip"]
    rawdata_hashes = {
        "kiwi-benchmark.zip": "sha256:c2c6171a4da720cd8801aed48c4199b4002adaf73a82976500c863b1dd0677b7"
    }
    table_schemas = {
        "timeseries": TIMESERIES_SCHEMA,
        "timeseries_metadata": TIMESERIES_METADATA_SCHEMA,
        "static_covariates": STATIC_COVARIATES_SCHEMA,
        "static_covariates_metadata": STATIC_COVARIATES_METADATA_SCHEMA,
    }

    def clean_table(self, key: Key, /) -> None:
        r"""Extract the preprocessed Parquet table from the bundled archive."""
        path = self.rawdata_paths["kiwi-benchmark.zip"]
        file = f"{key}.parquet"

        with ZipFile(path, "r") as archive:
            try:
                archive.extract(file, self.DATASET_DIR)
            except KeyError as exc:
                exc.add_note(f"Failed to extract table {key} from {path}")
                raise

    def load_table(self, key: Key, /) -> pl.DataFrame:
        r"""Load a preprocessed table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])

    def get_rawdata_file(self, fname: str, /) -> None:
        r"""Copy the bundled dataset archive to the raw-data directory."""
        self.LOGGER.info("Copying data files into %s.", self.rawdata_paths[fname])
        if __package__ is None:
            raise ValueError(f"Unexpected package: {__package__=}")
        with resources.path(__package__, fname) as path:
            shutil.copy(path, self.rawdata_paths[fname])
