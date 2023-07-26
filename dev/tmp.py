#!/usr/bin/env python
# +
# %config InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.
# %config InlineBackend.figure_format = 'svg'
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import logging

logging.basicConfig(level=logging.INFO)
# -

import tsdm

tsdm.datasets.timeseries.USHCN()

METADATA_DESCRIPTION = {
    # NOTE: extracted from parameter_types
    # fmt: off
    # "name" :
    "data" : [
        ("start_time"            , "timestamp", "datetime", None , None, None, None, None),
        ("end_time"              , "timestamp", "datetime", None , None, None, None, None),
        ("bioreactor_type_name"  , "string"   , "category", None , None, None, None, None),
        ("description"           , "string"   , "category", None , None, None, None, None),
        ("color"                 , "string"   , "category", None , None, None, None, None),
        ("medium_name"           , "string"   , "category", None , None, None, None, None),
        ("organism_name"         , "string"   , "category", None , None, None, None, None),
        ("plasmid_name"          , "string"   , "category", None , None, None, None, None),
        ("profile_name"          , "string"   , "category", None , None, None, None, None),
        ("Acetate_Dilution"      , "float32"  , "percent" , "%"  ,    0,  100, True, True),
        ("Feed_concentration_glc", "float32"  , "numeric" , "g/L",    0, None, True, None),
        ("Glucose_Dilution"      , "float32"  , "percent" , "%"  ,    0,  100, True, True),
        ("InducerConcentration"  , "float32"  , "numeric" , "mM" ,    0, None, True, None),
        ("OD_Dilution"           , "float32"  , "percent" , "%"  ,    0,  100, True, True),
        ("Stir_Max_Restarts"     , "float32"  , "numeric" , None ,    0, None, True, None),
        ("capacity_per_container", "float32"  , "numeric" , "ml" ,    0, None, True, None),
        ("pH_correction_factor"  , "float32"  , "fraction", "⅟₁" ,    0,    1, True, True),
        ("ph_Acid_conc"          , "float32"  , "numeric" , "mol",    0, None, True, None),
        ("ph_Base_conc"          , "float32"  , "numeric" , "mol",    0, None, True, None),
        ("ph_Ki"                 , "float32"  , "numeric" , None ,    0, None, True, None),
        ("ph_Kp"                 , "float32"  , "numeric" , None ,    0, None, True, None),
        ("ph_Tolerance"          , "float32"  , "numeric" , None ,    0, None, True, None),
    ],
    "columns" : ["name", "dtype", "kind", "unit", "lower_bound", "upper_bound", "lower_included", "upper_included"],
    "dtype": {
        # fmt: off
        "name"                          : "string",
        "dtype"                         : "string",
        "kind"                          : "string",
        "unit"                          : "string",
        "lower_bound"                   : "float32[pyarrow]",
        "upper_bound"                   : "float32[pyarrow]",
        "lower_included"                : "bool[pyarrow]",
        "upper_included"                : "bool[pyarrow]",
        # fmt: on
    }
    ,
    # fmt: on
}

from abc import abstractmethod

# +
from typing import Protocol


class Example(Protocol):
    def first(self) -> int:  # This is a protocol member
        return 42


# -

Example.first(object)

pd.DataFrame(METADATA_DESCRIPTION["data"])

pd.DataFrame(data)

tab = {
    "data": [
        ("Acetate", "float32", "fraction", "⅟₁", 0, 1, True, True),
        ("Base", "float32", "absolute", "uL", 0, None, True, None),
        (
            "Cumulated_feed_volume_glucose",
            "float32",
            "absolute",
            "uL",
            0,
            None,
            True,
            None,
        ),
        (
            "Cumulated_feed_volume_medium",
            "float32",
            "absolute",
            "uL",
            0,
            None,
            True,
            None,
        ),
        ("DOT", "float32", "percent", "%", 0, 100, True, True),
        ("Flow_Air", "float32", "absolute", "Ln/min", 0, None, True, None),
        ("Fluo_GFP", "float32", "absolute", "RFU", 0, None, True, None),
        ("Glucose", "float32", "absolute", "g/L", 0, None, True, None),
        ("InducerConcentration", "float32", "absolute", "mM", 0, None, True, None),
        ("OD600", "float32", "percent", "%", 0, 100, True, True),
        ("Probe_Volume", "float32", "absolute", "uL", 0, None, True, None),
        ("StirringSpeed", "float32", "absolute", "U/min", 0, 3200, True, True),
        ("Temperature", "float32", "linear", "°C", 0, 100, False, False),
        ("pH", "float32", "linear", "mL", 0, 14, False, False),
    ],
    "schema": {
        # fmt: off
        "name"                          : "string[pyarrow]",
        "dtype"                         : "string[pyarrow]",
        "kind"                          : "string[pyarrow]",
        "unit"                          : "string[pyarrow]",
        "lower_bound"                   : "float32[pyarrow]",
        "upper_bound"                   : "float32[pyarrow]",
        "lower_included"                : "bool[pyarrow]",
        "upper_included"                : "bool[pyarrow]",
        # fmt: on
    },
    "index": "name",
}
