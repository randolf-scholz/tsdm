r"""KIWI Run Data.

Extracted from iLab DataBase.
"""

__all__ = [
    # Classes
    "KIWI_RUNS",
    "KIWI",
]

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series

from tsdm.datasets.base import MultiFrameDataset, TimeSeriesCollection
from tsdm.utils import NULL_VALUES

__logger__ = logging.getLogger(__name__)


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
    "metadata_features" : {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "dtype" : "string[pyarrow]",
        "lower" : "Float32",
        "upper" : "Float32",
    },
    "value_features" : {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "dtype" : "string[pyarrow]",
        "lower" : "Float32",
        "upper" : "Float32",
    },
    "time_features": {
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


class KIWI_RUNS(MultiFrameDataset):
    r"""KIWI RUN Data.

    The cleaned data will consist of 3 parts:

    - timeseries
    - metadata
    - time_features
    - value_features
    - metadata_features

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

    BASE_URL = (
        "https://owncloud.innocampus.tu-berlin.de/index.php/s/fGFEJicrcjsxDBd/download/"
    )
    RAWDATA_HASH = r"dfed46bdcaa325434031fbd9f5d677d3679284246a81a9102671300db2d7f181"

    TABLE_HASH = {
        "timeseries": 7423431600366696406,
        "metadata": 2037390744336738142,
        "time_features": 4775909302393294764,
        "value_features": -6386491732663357532,
        "metadata_features": 4215379263850919231,
    }

    rawdata_files = "kiwi_experiments.pk"
    rawdata_paths: Path

    timeseries: DataFrame
    metadata: DataFrame

    time_features: DataFrame
    value_features: DataFrame
    metadata_features: DataFrame

    index_features = None
    global_metadata = None
    global_features = None

    KEYS = [
        "timeseries",
        "metadata",
        "time_features",
        "value_features",
        "metadata_features",
    ]

    @property
    def index(self) -> Index:
        r"""Return the index of the dataset."""
        return self.metadata.index

    def clean_table(self, key):
        if key in ["timeseries", "value_features"]:
            self.clean_timeseries()
        if key in ["index", "metadata", "time_features", "metadata_features"]:
            self.clean_metadata()

    def clean_metadata(self):
        r"""Clean metadata."""
        # load rawdata
        with open(self.rawdata_paths, "rb") as file:
            self.LOGGER.info("Loading raw data from %s", self.rawdata_paths)
            data = pickle.load(file)

        # generate dataframe
        metadata_dict = {
            (outer_key, inner_key): tables["metadata"]
            for outer_key, experiment in data.items()
            for inner_key, tables in experiment.items()
        }
        metadata = pd.concat(metadata_dict, names=["run_id", "exp_id"])
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

        # time_features
        time_features = metadata[["start_time", "end_time"]].copy()
        time_features["lower"] = (
            time_features["start_time"] - time_features["start_time"]
        )
        time_features["upper"] = time_features["end_time"] - time_features["start_time"]
        time_features["unit"] = "s"  # time is rounded to seconds
        time_features["dtype"] = "datetime64[ns]"
        time_features["scale"] = "linear"
        time_features = time_features[
            ["unit", "scale", "dtype", "lower", "upper", "start_time", "end_time"]
        ]
        time_features = time_features.astype(column_dtypes["time_features"])

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

        # fmt: off
        metadata_features_dict = {
            # column                   [unit,  scale,      dtype, lower , upper]
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

        metadata_features = DataFrame.from_dict(
            metadata_features_dict,
            orient="index",
            columns=column_dtypes["metadata_features"],
        )
        metadata_features["dtype"] = metadata.dtypes.astype("string[pyarrow]")
        metadata_features = metadata_features[
            ["unit", "scale", "dtype", "lower", "upper"]
        ]
        metadata_features = metadata_features.astype(column_dtypes["metadata_features"])

        # Remove values out of bounds
        for col in metadata:
            lower: Series = metadata_features.loc[col, "lower"]
            upper: Series = metadata_features.loc[col, "upper"]
            value = metadata[col]
            mask = (lower > value) | (value > upper)
            if mask.any():
                print(
                    f"Removing {mask.mean():.2%} of data that does not match {col} bounds"
                )
                metadata.loc[mask, col] = pd.NA

        # Finalize tables
        time_features.to_parquet(self.dataset_paths["time_features"])
        metadata = metadata.dropna(how="all")
        metadata.to_parquet(self.dataset_paths["metadata"])
        metadata_features.to_parquet(self.dataset_paths["metadata_features"])

    def clean_timeseries(self):
        r"""Clean timeseries data and save to disk."""
        # load rawdata
        with open(self.rawdata_paths, "rb") as file:
            self.LOGGER.info("Loading raw data from %s", self.rawdata_paths)
            data = pickle.load(file)

        # Generate DataFrame
        timeseries_dict = {
            (outer_key, inner_key): tables["measurements_aggregated"]
            for outer_key, experiment in data.items()
            for inner_key, tables in experiment.items()
        }

        timeseries = pd.concat(
            timeseries_dict, names=["run_id", "exp_id"], verify_integrity=True
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
        for col in timeseries:
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

        # Timeseries Features
        # fmt: off
        value_features_dict = {
            "Acetate"                       : ["%",      "percent",  pd.NA, 0,   100      ],
            "Base"                          : ["uL",     "absolute", pd.NA, 0,   np.inf   ],
            "Cumulated_feed_volume_glucose" : ["uL",     "absolute", pd.NA, 0,   np.inf   ],
            "Cumulated_feed_volume_medium"  : ["uL",     "absolute", pd.NA, 0,   np.inf   ],
            "DOT"                           : ["%",      "percent",  pd.NA, 0,   100      ],
            "Flow_Air"                      : ["Ln/min", "absolute", pd.NA, 0,   np.inf   ],
            "Fluo_GFP"                      : ["RFU",    "absolute", pd.NA, 0,   1_000_000],
            "Glucose"                       : ["g/L",    "absolute", pd.NA, 0,   20       ],
            "InducerConcentration"          : ["mM",     "absolute", pd.NA, 0,   np.inf   ],
            "OD600"                         : ["%",      "percent",  pd.NA, 0,   100      ],
            "Probe_Volume"                  : ["uL",     "absolute", pd.NA, 0,   np.inf   ],
            "StirringSpeed"                 : ["U/min",  "absolute", pd.NA, 0,   np.inf   ],
            "Temperature"                   : ["°C",     "linear",   pd.NA, 20,  45       ],
            "Volume"                        : ["mL",     "absolute", pd.NA, 0,   np.inf   ],
            "pH"                            : ["pH",     "linear",   pd.NA, 4,   10       ],
        }
        # fmt: on

        value_features = DataFrame.from_dict(
            value_features_dict,
            orient="index",
            columns=column_dtypes["value_features"],
        )
        value_features["dtype"] = timeseries.dtypes.astype("string[pyarrow]")
        value_features = value_features.astype(column_dtypes["value_features"])

        # Remove values out of bounds
        for col in timeseries:
            lower: Series = value_features.loc[col, "lower"]
            upper: Series = value_features.loc[col, "upper"]
            value = timeseries[col]
            mask = (lower > value) | (value > upper)
            if mask.any():
                print(
                    f"Removing {mask.mean():.2%} of data that does not match {col} bounds"
                )
                timeseries.loc[mask, col] = pd.NA

        # Remove data outside of time bounds
        ts = timeseries.reset_index("measurement_time")
        ts = ts.join(self.time_features[["start_time", "end_time"]])
        time = ts["measurement_time"]
        mask = (ts["start_time"] <= time) & (time <= ts["end_time"])
        print(f"Removing {(~mask).mean():.2%} of data that does not match tmin/tmax")
        ts = ts[mask]
        ts["measurement_time"] = ts["measurement_time"] - ts["start_time"]
        ts = ts.set_index("measurement_time", append=True)
        timeseries = ts[timeseries.columns]

        # Aggregate Measurements (non-destructive)
        # https://stackoverflow.com/questions/74115705
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
        )

        # Finalize Tables
        value_features.to_parquet(self.dataset_paths["value_features"])
        timeseries = timeseries.dropna(how="all")
        timeseries = timeseries.sort_values(["run_id", "exp_id", "measurement_time"])
        timeseries.to_parquet(self.dataset_paths["timeseries"])


class KIWI(TimeSeriesCollection):
    """The KIWI dataset."""

    def __init__(self):
        ds = KIWI_RUNS()

        super().__init__(
            index=ds.index,
            timeseries=ds.timeseries,
            metadata=ds.metadata,
            global_metadata=ds.global_metadata,
            index_features=ds.index_features,
            time_features=ds.time_features,
            value_features=ds.value_features,
            metadata_features=ds.metadata_features,
            global_features=None,
        )


# INFO:tsdm.datasets.kiwi_runs.KIWI_RUNS:Adding keys as attributes.
# timeseries '7423431600366696406'
# 'metadata.parquet' '2037390744336738142'
#  'time_features.parquet''4775909302393294764'
# 'value_features.parquet''-6386491732663357532'
# 'metadata_features.parquet'  '4215379263850919231'
# 'filehash='4215379263850919231''.
