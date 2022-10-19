r"""KIWI Run Data.

Extracted from iLab DataBase.
"""

__all__ = [
    # Classes
    "KIWI_RUNS",
]

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series

from tsdm.datasets.base import MultiFrameDataset
from tsdm.utils import NULL_VALUES

__logger__ = logging.getLogger(__name__)


# fmt: off
column_dtypes = {
    "metadata": {
        # "experiment_id"          : "Int32",
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
    "setpoints" : {
        "cultivation_age"             : "Int32",
        "setpoint_id"                 : "Int32",
        "unit"                        : "string[pyarrow]",
        "Feed_glc_cum_setpoints"      : "Int32",
        "Flow_Air"                    : "Int32",
        "InducerConcentration"        : "Float32",
        "Puls_AceticAcid"             : "Int32",
        "Puls_Glucose"                : "Int32",
        "Puls_Medium"                 : "Int32",
        "StirringSpeed"               : "Int32",
        "pH"                          : "Float32",
        "Flow_Nitrogen"               : "Int32",
        "Flow_O2"                     : "Int32",
        "Feed_dextrine_cum_setpoints" : "Int32",
        "Temperature"                 : "Int32",
    },
    "metadata_features" : {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "lower" : "Float32",
        "upper" : "Float32",
    },
    "timeseries_features" : {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "lower" : "Float32",
        "upper" : "Float32",
    },
    "setpoints_features" : {
        "unit"  : "string[pyarrow]",
        "scale" : "string[pyarrow]",
        "lower" : "Float32",
        "upper" : "Float32",
    },
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
    - tmin, tmax
    - globals
    - timeseries_features
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
    RAWDATA_SHA256 = r"dfed46bdcaa325434031fbd9f5d677d3679284246a81a9102671300db2d7f181"
    rawdata_files = "kiwi_experiments.pk"
    rawdata_paths: Path

    # RAWDATA_SHA256 = "79d8d15069b4adc6d77498472008bd87e3699a75bb612029232bd051ecdbb078"
    # DATASET_SHA256 = {
    #     "timeseries": "819d5917c5ed65cec7855f02156db1abb81ca3286e57533ee15eb91c072323f9",
    #     "metadata": "8b4d3f922c2fb3988ae606021492aa10dd3d420b3c6270027f91660a909429ae",
    #     "units": "aa4d0dd22e0e44c78e7034eb49ed39cde371fa1e4bf9b9276e9e2941c54e5eca",
    # }
    # DATASET_SHAPE = {
    #     "timeseries": (386812, 15),
    #     "metadata": (264, 11),
    # }
    tmin: Series
    tmax: Series
    timeseries: DataFrame
    metadata: DataFrame
    timeseries_features: DataFrame
    metadata_features: DataFrame
    KEYS = [
        "tmin",
        "tmax",
        "timeseries",
        "metadata",
        "timeseries_features",
        "metadata_features",
    ]

    @property
    def index(self) -> Index:
        return self.metadata.index

    def clean_table(self, key):
        if key in ["timeseries", "timeseries_features"]:
            self.clean_timeseries()
        if key in ["index", "tmin", "tmax", "metadata", "metadata_features"]:
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

        # select columns
        tmin = metadata["start_time"]
        tmax = metadata["end_time"]
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
            # column                   [unit, scale, lower bound, upper bound]
            "bioreactor_id"          : [pd.NA, "category", pd.NA, pd.NA ],
            "container_number"       : [pd.NA, "category", pd.NA, pd.NA ],
            "color"                  : [pd.NA, "category", pd.NA, pd.NA ],
            "profile_name"           : [pd.NA, "category", pd.NA, pd.NA ],
            "plasmid_id"             : [pd.NA, "category", pd.NA, pd.NA ],
            "Feed_concentration_glc" : ["g/L", "absolute", pd.NA, pd.NA ],
            "OD_Dilution"            : ["%",   "percent",  0,     100   ],
            "pH_correction_factor"   : [pd.NA, "factor",   0,     np.inf],
            "ph_Tolerance"           : [pd.NA, "linear",   0,     np.inf],
            "μ_set"                  : ["%",   "percent",  0,     100   ],
            "IPTG"                   : ["mM",  "absolute", 0,     np.inf],
        }
        # fmt: on

        metadata_features = DataFrame.from_dict(
            metadata_features_dict,
            orient="index",
            columns=column_dtypes["metadata_features"],
        )
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
        tmin.to_parquet(self.dataset_paths["tmin"])
        tmax.to_parquet(self.dataset_paths["tmax"])
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
        timeseries_features_dict = {
            "Acetate"                       : ["%",      "percent",   0,   100      ],
            "Base"                          : ["uL",     "absolute",  0,   np.inf   ],
            "Cumulated_feed_volume_glucose" : ["uL",     "absolute",  0,   np.inf   ],
            "Cumulated_feed_volume_medium"  : ["uL",     "absolute",  0,   np.inf   ],
            "DOT"                           : ["%",      "percent",   0,   100      ],
            "Flow_Air"                      : ["Ln/min", "absolute",  0,   np.inf   ],
            "Fluo_GFP"                      : ["RFU",    "absolute",  0,   1_000_000],
            "Glucose"                       : ["g/L",    "absolute",  0,   20       ],
            "InducerConcentration"          : ["mM",     "absolute",  0,   np.inf   ],
            "OD600"                         : ["%",      "percent",   0,   100      ],
            "Probe_Volume"                  : ["uL",     "absolute",  0,   np.inf   ],
            "StirringSpeed"                 : ["U/min",  "absolute",  0,   np.inf   ],
            "Temperature"                   : ["°C",     "linear",    20,  45       ],
            "Volume"                        : ["mL",     "absolute",  0,   np.inf   ],
            "pH"                            : ["pH",     "linear",    4,   10       ],
        }
        # fmt: on

        timeseries_features = DataFrame.from_dict(
            timeseries_features_dict,
            orient="index",
            columns=column_dtypes["timeseries_features"],
        )
        timeseries_features = timeseries_features.astype(
            column_dtypes["timeseries_features"]
        )

        # Remove values out of bounds
        for col in timeseries:
            lower: Series = timeseries_features.loc[col, "lower"]
            upper: Series = timeseries_features.loc[col, "upper"]
            value = timeseries[col]
            mask = (lower > value) | (value > upper)
            if mask.any():
                print(
                    f"Removing {mask.mean():.2%} of data that does not match {col} bounds"
                )
                timeseries.loc[mask, col] = pd.NA

        # Remove data outside of time bounds
        ts = timeseries.reset_index("measurement_time")
        ts = ts.join([self.tmin, self.tmax])
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

        # Finalize Table
        timeseries = timeseries.dropna(how="all")
        timeseries = timeseries.sort_values(["run_id", "exp_id", "measurement_time"])
        timeseries.to_parquet(self.dataset_paths["timeseries"])
        timeseries_features.to_parquet(self.dataset_paths["timeseries_features"])
