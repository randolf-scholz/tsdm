r"""KIWI Run Data.

Extracted from iLab DataBase.
"""

__all__ = [
    # Classes
    "KIWI_RUNS",
    "KiwiDataset",
]

import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from tsdm.datasets.base import MultiTableDataset, TimeSeriesCollection
from tsdm.utils.constants import NULL_VALUES

# fmt: off
column_dtypes = {
    "metadata": {
        # "experiment_id"        : "Int32",
        "bioreactor_id"          : "Int32",
        "container_number"       : "Int32",
        "profile_id"             : "Int32",
        "starter_culture_id"     : "Int32",
        "inactivation_method_id" : "Int32",
        "description_x"          : "string[pyarrow]",
        "color"                  : "string[pyarrow]",
        "profile_name"           : "string[pyarrow]",
        "folder_id_x"            : "Int32",
        "organism_id"            : "Int32",
        "plasmid_id"             : "Int32",
        "medium_id"              : "Int32",
        "description_y"          : "string[pyarrow]",
        # "run_id"                 : "Int32",
        "Acetate_Dilution"       : "Int32",
        "Feed_concentration_glc" : "Int32",
        "InducerConcentration"   : "Int32",
        "OD_Dilution"            : "Int32",
        "Stir_Max_Restarts"      : "Int32",
        "pH_correction_factor"   : "Float32",
        "ph_Base_conc"           : "Float32",
        "ph_Ki"                  : "Float32",
        "ph_Kp"                  : "Float32",
        "ph_Tolerance"           : "Float32",
        "run_name"               : "string[pyarrow]",
        "folder_id_y"            : "Int32",
        "pms_id"                 : "Int32",
        "status_id"              : "Int32",
        "start_time"             : "datetime64[ns]",
        "end_time"               : "datetime64[ns]",
        "description"            : "string[pyarrow]",
        "conclusion"             : "string[pyarrow]",
        "container_label"        : "string[pyarrow]",
        "is_template"            : "Int32",
        "Glucose_Dilution"       : "Int32",
        "ph_Acid_conc"           : "Int32",
        # added columns
        "μ_set"                  : "Int32",
        "IPTG"                   : "Float32",
    },
    "timeseries": {
        # "measurement_time"              : "datetime64[ns]",
        "unit"                            : "string[pyarrow]",
        "Flow_Air"                        : "Int32",
        "Flow_Nitrogen"                   : "Int32",
        "Flow_O2"                         : "Int32",
        "StirringSpeed"                   : "Int32",
        "Temperature"                     : "Float32",
        "Acetate"                         : "Float32",
        "Base"                            : "Int32",
        "Cumulated_feed_volume_glucose"   : "Int32",
        "Cumulated_feed_volume_medium"    : "Float32",
        "DOT"                             : "Float32",
        "Fluo_GFP"                        : "Float32",
        "Glucose"                         : "Float32",
        "OD600"                           : "Float32",
        "Probe_Volume"                    : "Int32",
        "pH"                              : "Float32",
        "InducerConcentration"            : "Float32",
        "remark"                          : "string",
        "Volume"                          : "Float32",
        "Acid"                            : "Int32",
    },
    # "setpoints" : {
    #     "cultivation_age"             : "Int32",
    #     "setpoint_id"                 : "Int32",
    #     "unit"                        : "string[pyarrow]",
    #     "Feed_glc_cum_setpoints"      : "Int32",
    #     "Flow_Air"                    : "Int32",
    #     "InducerConcentration"        : "Float32",
    #     "Puls_AceticAcid"             : "Int32",
    #     "Puls_Glucose"                : "Int32",
    #     "Puls_Medium"                 : "Int32",
    #     "StirringSpeed"               : "Int32",
    #     "pH"                          : "Float32",
    #     "Flow_Nitrogen"               : "Int32",
    #     "Flow_O2"                     : "Int32",
    #     "Feed_dextrine_cum_setpoints" : "Int32",
    #     "Temperature"                 : "Int32",
    # },
    "metadata_description" : {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "dtype" : "string[pyarrow]",
        "lower" : "Float32",
        "upper" : "Float32",
    },
    "timeseries_description" : {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "dtype" : "string[pyarrow]",
        "lower" : "Float32",
        "upper" : "Float32",
    },
    "timeindex_description": {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "dtype" : "string[pyarrow]",
        "lower" : "timedelta64[ns]",
        "upper" : "timedelta64[ns]",
        "start_time" : "datetime64[ns]",
        "end_time"   : "datetime64[ns]",
    },
    # "setpoints_features" : {
    #     "unit"  : "string[pyarrow]",
    #     "scale" : "string[pyarrow]",
    #     "dtype": "string[pyarrow]",
    #     "lower" : "Float32",
    #     "upper" : "Float32",
    # },
}
# fmt: on

# fmt: off
selected_columns = {
    "metadata" : {
        # "experiment_id"          : True,
        "bioreactor_id"          : True,
        "container_number"       : True,
        "profile_id"             : False,
        "description_x"          : False,
        "color"                  : True,
        "profile_name"           : True,
        "plasmid_id"             : True,
        # "run_id"                 : True,
        "Feed_concentration_glc" : True,
        "OD_Dilution"            : True,
        "pH_correction_factor"   : True,
        "ph_Tolerance"           : True,
        "run_name"               : False,
        "folder_id_y"            : False,
        "start_time"             : False,
        "end_time"               : False,
        "description"            : False,
        "μ_set"                  : True,
        "IPTG"                   : True,
    },
    "setpoints": {
        "cultivation_age"             : False,
        "setpoint_id"                 : False,
        "unit"                        : False,
        "Feed_glc_cum_setpoints"      : True,
        "Flow_Air"                    : True,
        "InducerConcentration"        : True,
        "Puls_AceticAcid"             : False,
        "Puls_Glucose"                : False,
        "Puls_Medium"                 : False,
        "StirringSpeed"               : True,
        "pH"                          : True,
        "Flow_Nitrogen"               : False,
        "Flow_O2"                     : False,
        "Feed_dextrine_cum_setpoints" : False,
        "Temperature"                 : False,
    },
    "timeseries" :  {
        "unit"                          : False,
        "Flow_Air"                      : True,
        "Flow_Nitrogen"                 : False,
        "Flow_O2"                       : False,
        "StirringSpeed"                 : True,
        "Temperature"                   : True,
        "Acetate"                       : True,
        "Base"                          : True,
        "Cumulated_feed_volume_glucose" : True,
        "Cumulated_feed_volume_medium"  : True,
        "DOT"                           : True,
        "Fluo_GFP"                      : True,
        "Glucose"                       : True,
        "OD600"                         : True,
        "Probe_Volume"                  : True,
        "pH"                            : True,
        "InducerConcentration"          : True,
        "remark"                        : False,
        "Volume"                        : False,
        "Acid"                          : False,
    }
}
# fmt: on

null_values = NULL_VALUES + [
    "value written to setpoints has been transferred to this table."
]

# Timeseries Features
# fmt: off
timeseries_description_dict = {
    # Name                          : [Unit,     Type,      Dtype, Lower, Upper, Lower included, Upper included]
    "Acetate"                       : ["%",      "percent",  pd.NA, 0,   100      , True, True],
    "Base"                          : ["uL",     "absolute", pd.NA, 0,   np.inf   , True, True],
    "Cumulated_feed_volume_glucose" : ["uL",     "absolute", pd.NA, 0,   np.inf   , True, True],
    "Cumulated_feed_volume_medium"  : ["uL",     "absolute", pd.NA, 0,   np.inf   , True, True],
    "DOT"                           : ["%",      "percent",  pd.NA, 0,   100      , True, True],
    "Flow_Air"                      : ["Ln/min", "absolute", pd.NA, 0,   np.inf   , True, True],
    "Fluo_GFP"                      : ["RFU",    "absolute", pd.NA, 0,   1_000_000, True, True],
    "Glucose"                       : ["g/L",    "absolute", pd.NA, 0,   20       , True, True],
    "InducerConcentration"          : ["mM",     "absolute", pd.NA, 0,   np.inf   , True, True],
    "OD600"                         : ["%",      "percent",  pd.NA, 0,   100      , True, True],
    "Probe_Volume"                  : ["uL",     "absolute", pd.NA, 0,   np.inf   , True, True],
    "StirringSpeed"                 : ["U/min",  "absolute", pd.NA, 0,   np.inf   , True, True],
    "Temperature"                   : ["°C",     "linear",   pd.NA, 20,  45       , True, True],
    "Volume"                        : ["mL",     "absolute", pd.NA, 0,   np.inf   , True, True],
    "pH"                            : ["pH",     "linear",   pd.NA, 4,   10       , True, True],
}
# fmt: on

# fmt: off
metadata_description_dict = {
    # Name                   : [Unit,     Type,      Dtype, Lower, Upper, Lower included, Upper included]
    "bioreactor_id"          : [pd.NA, "category", pd.NA, pd.NA, pd.NA ],
    "container_number"       : [pd.NA, "category", pd.NA, pd.NA, pd.NA ],
    "color"                  : [pd.NA, "category", pd.NA, pd.NA, pd.NA ],
    "profile_name"           : [pd.NA, "category", pd.NA, pd.NA, pd.NA ],
    "plasmid_id"             : [pd.NA, "category", pd.NA, pd.NA, pd.NA ],
    "Feed_concentration_glc" : ["g/L", "absolute", pd.NA, pd.NA, pd.NA ],
    "OD_Dilution"            : ["%",   "percent",  pd.NA, 0,     100   ],
    "pH_correction_factor"   : [pd.NA, "factor",   pd.NA, 0,     np.inf],
    "ph_Tolerance"           : [pd.NA, "linear",   pd.NA, 0,     np.inf],
    "μ_set"                  : ["%",   "percent",  pd.NA, 0,     100   ],
    "IPTG"                   : ["mM",  "absolute", pd.NA, 0,     np.inf],
}
# fmt: on


class KIWI_RUNS(MultiTableDataset):
    r"""KIWI RUN Data.

    The cleaned data will consist of 3 parts:

    - timeseries
    - metadata
    - timeseries_description
    - timeseries_description
    - metadata_description

    Rawdata Format:

    .. code-block:: python

        dict[
            int,  # run_id
            dict[
                int,  # experiment_id
                dict[
                    "metadata",
                    :DataFrame,  # static
                    "setpoints":DataFrame,  # static
                    "measurements_reactor",
                    :DataFrame,  # TimeTensor
                    "measurements_array",
                    :DataFrame,  # TimeTensor
                    "measurements_aggregated":DataFrame,  # TimeTensor
                ],
            ],
        ]
    """

    BASE_URL = (
        "https://owncloud.innocampus.tu-berlin.de/index.php/s/fGFEJicrcjsxDBd/download/"
    )

    table_names = [
        "timeseries",
        "metadata",
        "timeseries_description",
        "metadata_description",
        "timeindex_description",
    ]

    table_hashes = {
        "timeseries": "pandas:BUOOC5Z2OKCIW",
        "metadata": "pandas:PRI8003H72CW",
        "timeseries_description": "pandas:WHK7T51NT1HS",
        "metadata_description": "pandas:6A8LIT5JTX99",
        "timeindex_description": "pandas:BAKJST6Y68SAM",
    }

    rawdata_files = ["kiwi_experiments.pk"]
    rawdata_hashes = {
        "kiwi_experiments.pk": "sha256:dfed46bdcaa325434031fbd9f5d677d3679284246a81a9102671300db2d7f181",
    }

    timeseries: DataFrame
    metadata: DataFrame
    timeindex_description: DataFrame
    timeseries_description: DataFrame
    metadata_description: DataFrame
    metaindex_description = None

    @property
    def index(self) -> Index:
        r"""Return the index of the dataset."""
        return self.metadata.index

    def clean_table(self, key: str) -> None:
        if key in ["timeseries", "timeseries_description"]:
            self.clean_timeseries()
        if key in [
            # "index",
            "metadata",
            "metadata_description",
            "timeindex_description",
        ]:
            self.clean_metadata()

    def clean_metadata(self) -> None:
        r"""Clean metadata."""
        # load rawdata
        self.LOGGER.info("Loading raw data from %s", self.rawdata_paths)
        with open(self.rawdata_paths["kiwi_experiments.pk"], "rb") as file:
            data = pickle.load(file)

        # generate dataframe
        metadata_dict = {
            (outer_key, inner_key): tables["metadata"]
            for outer_key, experiment in data.items()
            for inner_key, tables in experiment.items()
        }
        metadata = pd.concat(metadata_dict, names=["run_id", "experiment_id"])
        metadata = metadata.reset_index(-1, drop=True)
        metadata = metadata.drop(columns=["run_id", "experiment_id"])

        # generate μ-set columns
        mu_sets = metadata["description_x"].str.split(" ", expand=True)
        mu_sets = mu_sets.astype("string[pyarrow]")
        mu_sets.columns = ["name", "percent", "amount", "unit", "chemical"]
        mu_sets["percent"] = mu_sets["percent"].str.split("%", expand=True)[0]
        metadata["μ_set"] = mu_sets["percent"]
        metadata["IPTG"] = mu_sets["amount"]

        # fix datatypes
        metadata = metadata.astype(column_dtypes["metadata"])

        # timeseries_description
        timeindex_description = metadata[["start_time", "end_time"]].copy()
        timeindex_description["lower"] = (
            timeindex_description["start_time"] - timeindex_description["start_time"]
        )
        timeindex_description["upper"] = (
            timeindex_description["end_time"] - timeindex_description["start_time"]
        )
        timeindex_description["unit"] = "s"  # time is rounded to seconds
        timeindex_description["dtype"] = "datetime64[ns]"
        timeindex_description["scale"] = "linear"
        timeindex_description = timeindex_description[
            ["unit", "scale", "dtype", "lower", "upper", "start_time", "end_time"]
        ]
        timeindex_description = timeindex_description.astype(
            column_dtypes["timeindex_description"]
        )

        # select columns
        columns = [key for key, val in selected_columns["metadata"].items() if val]
        metadata = metadata[columns]

        # Metadata Features
        units = {}
        mask = mu_sets["amount"].notna()
        mu_set_unit = list(mu_sets["unit"].loc[mask].unique())
        assert len(mu_set_unit) == 1
        units["IPTG"] = mu_set_unit[0]
        units["μ_set"] = "%"

        metadata_description = DataFrame.from_dict(
            metadata_description_dict,
            orient="index",
            columns=column_dtypes["metadata_description"],
        )
        metadata_description["dtype"] = metadata.dtypes.astype("string[pyarrow]")
        metadata_description = metadata_description[
            ["unit", "scale", "dtype", "lower", "upper"]
        ]
        metadata_description = metadata_description.astype(
            column_dtypes["metadata_description"]
        )

        # Remove values out of bounds
        for col in metadata.columns:
            lower: Series = metadata_description.loc[col, "lower"]
            upper: Series = metadata_description.loc[col, "upper"]
            value = metadata[col]
            mask = (lower > value) | (value > upper)
            if mask.any():
                print(
                    f"Removing {mask.mean():8.3%} of data that does not match {col} bounds"
                )
                metadata.loc[mask, col] = pd.NA
        metadata = metadata.dropna(how="all")

        # Serialize tables
        self.LOGGER.info("Serializing metadata tables.")
        self.serialize(
            timeindex_description, self.dataset_paths["timeindex_description"]
        )
        self.serialize(metadata, self.dataset_paths["metadata"])
        self.serialize(metadata_description, self.dataset_paths["metadata_description"])

    def clean_timeseries(self) -> None:
        r"""Clean timeseries data and save to disk."""
        # load rawdata
        self.LOGGER.info("Loading raw data from %s", self.rawdata_paths)
        with open(self.rawdata_paths["kiwi_experiments.pk"], "rb") as file:
            data = pickle.load(file)

        # Generate DataFrame
        timeseries_dict = {
            (outer_key, inner_key): tables["measurements_aggregated"]
            for outer_key, experiment in data.items()
            for inner_key, tables in experiment.items()
        }

        timeseries: DataFrame = pd.concat(
            timeseries_dict, names=["run_id", "experiment_id"], verify_integrity=True
        )
        timeseries = timeseries.reset_index(-1, drop=True)
        timeseries = timeseries.set_index("measurement_time", append=True)

        # fix data types
        timeseries = timeseries.astype(column_dtypes["timeseries"])

        # replace spurious na values
        timeseries["unit"].replace("-", pd.NA, inplace=True)

        # Select columns
        timeseries_units = timeseries["unit"]
        timeseries = timeseries.drop(columns=["unit"])

        # remove non-informative columns:
        # columns with single value carry no information
        mask: Series = timeseries.nunique() > 1
        # only keep columns that appear in at least half of the runs
        mask &= (timeseries.groupby("run_id").nunique() > 0).mean() > 0.5
        timeseries = timeseries[timeseries.columns[mask]]

        # Validate units
        assert all(timeseries.notna().sum(axis=1) <= 1), "multiple measurements!"

        units = {}
        for col in timeseries.columns:
            mask = timeseries[col].notna()
            units[col] = list(timeseries_units.loc[mask].unique())
            assert len(units[col]) == 1, f"Multiple different units in {col}!"

        units = Series({k: v[0] for k, v in units.items()}, dtype="string[pyarrow]")
        units[["Acetate", "OD600", "DOT", "pH"]] = ["%", "%", "%", "pH"]

        # Check that data is non-trivial
        uniques_per_run_id = timeseries.groupby("run_id").nunique()
        assert ((uniques_per_run_id > 1).sum() > 1).all()

        # Select Columns
        columns = [key for key, val in selected_columns["timeseries"].items() if val]
        timeseries = timeseries[columns]

        timeseries_description = DataFrame.from_dict(
            timeseries_description_dict,
            orient="index",
            columns=column_dtypes["timeseries_description"],
        )
        timeseries_description["dtype"] = timeseries.dtypes.astype("string[pyarrow]")
        timeseries_description = timeseries_description.astype(
            column_dtypes["timeseries_description"]
        )
        timeseries_description = timeseries_description.loc[timeseries.columns]

        # Remove values out of bounds
        for col in timeseries.columns:
            lower: Series = timeseries_description.loc[col, "lower"]
            upper: Series = timeseries_description.loc[col, "upper"]
            value = timeseries[col]
            mask = (lower > value) | (value > upper)
            if mask.any():
                print(
                    f"Removing {mask.mean():8.3%} of data that does not match {col} bounds"
                )
                timeseries.loc[mask, col] = pd.NA

        # Remove data outside of time bounds
        ts = timeseries.reset_index("measurement_time")
        ts = ts.join(self.timeindex_description[["start_time", "end_time"]])
        time = ts["measurement_time"]
        mask = (ts["start_time"] <= time) & (time <= ts["end_time"])
        print(f"Removing {(~mask).mean():8.3%} of data that does not match tmin/tmax")
        ts = ts[mask]
        ts["measurement_time"] = ts["measurement_time"] - ts["start_time"]
        ts = ts.set_index("measurement_time", append=True)
        timeseries = ts[timeseries.columns]

        # Aggregate Measurements (non-destructive)
        # TODO: https://stackoverflow.com/questions/74115705
        # TODO: is there a way to do it without stacking?
        ts = timeseries.stack().to_frame(name="val")
        counts = ts.groupby(level=[0, 1, 2, 3]).cumcount()
        timeseries = (
            ts.set_index(counts, append=True)
            .loc[:, "val"]
            .unstack(level=3)
            .reindex(timeseries.columns, axis=1)
            .reset_index(level=3, drop=True)
            .astype(timeseries.dtypes)
            .dropna(how="all")
            .sort_values(["run_id", "experiment_id", "measurement_time"])
        )

        # Serialize Tables
        self.LOGGER.info("Serializing tiemseries tables.")
        self.serialize(timeseries, self.dataset_paths["timeseries"])
        self.serialize(
            timeseries_description, self.dataset_paths["timeseries_description"]
        )


class KiwiDataset(TimeSeriesCollection):
    r"""The KIWI dataset."""

    metaindex: MultiIndex
    timeseries: DataFrame
    metadata: DataFrame
    global_metadata: None
    metaindex_description: DataFrame
    timeindex_description: DataFrame
    timeseries_description: DataFrame
    metadata_description: DataFrame
    global_metadata_description: None

    def __init__(self) -> None:
        ds = KIWI_RUNS()

        super().__init__(
            metaindex=ds.index,
            timeseries=ds.timeseries,
            metadata=ds.metadata,
            global_metadata=None,
            metaindex_description=ds.metaindex_description,
            timeindex_description=ds.timeindex_description,
            timeseries_description=ds.timeseries_description,
            metadata_description=ds.metadata_description,
            global_metadata_description=None,
        )


# INFO:tsdm.datasets.kiwi_runs.KIWI_RUNS:Adding keys as attributes.
# timeseries '7423431600366696406'
# 'metadata.parquet' '2037390744336738142'
#  'timeseries_description.parquet''4775909302393294764'
# 'timeseries_description.parquet''-6386491732663357532'
# 'metadata_description.parquet'  '4215379263850919231'
# 'filehash='4215379263850919231''.
