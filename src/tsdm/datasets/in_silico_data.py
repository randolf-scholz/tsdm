r"""In silico experiments."""

__all__ = ["InSilicoData"]

import shutil
from importlib import resources
from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame

from tsdm.datasets import examples
from tsdm.datasets.base import SingleTableDataset
from tsdm.utils.data import remove_outliers


class InSilicoData(SingleTableDataset[DataFrame]):
    r"""Artificially generated data, 8 runs, 7 attributes, ~465 samples.

    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    |         | Time    | Biomass | Substrate | Acetate | DOTm  | Product | Volume    | Feed |
    +=========+=========+=========+===========+=========+=======+=========+===========+======+
    | unit    | float   | g/L     | g/l       | g/L     | %     | g/L     | L         | µL   |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    | domain  | [0, 12] | >0      | >0        | >0      | [0,1] | >0      | [0, 0.01] | >0   |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    | missing | -       | 99%     | 99%       | 99%     | 12%   | 99%     | 93%       | -    |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    """

    rawdata_files = ["in_silico.zip"]
    rawdata_hashes = {
        "in_silico.zip": "sha256:ee9ad6278fb27dd933c22aecfc7b5b2501336e859a7f012cace2bb265f713cba",
    }
    table_shapes = {"timeseries": (5206, 7)}
    table_schemas = {
        "timeseries": {
            # fmt: off
            "Biomass"   : "float[pyarrow]",
            "Substrate" : "float[pyarrow]",
            "Acetate"   : "float[pyarrow]",
            "DOTm"      : "float[pyarrow]",
            "Product"   : "float[pyarrow]",
            "Volume"    : "float[pyarrow]",
            "Feed"      : "float[pyarrow]",
            # fmt: on
        },
        "timeseries_description": {
            # fmt: off
            "variable"       : "string[pyarrow]",
            "lower"          : "float32[pyarrow]",
            "upper"          : "float32[pyarrow]",
            "lower_included" : "bool[pyarrow]",
            "upper_included" : "bool[pyarrow]",
            "unit"           : "string[pyarrow]",
            "description"    : "string[pyarrow]",
            # fmt: on
        },
    }

    def _timeseries_description(self) -> DataFrame:
        data = [
            # fmt: off
            ("Biomass"  , 0, None, True, True, "g/L", None),
            ("Substrate", 0, None, True, True, "g/L", None),
            ("Acetate"  , 0, None, True, True, "g/L", None),
            ("DOTm"     , 0, 1,    True, True, "%",   None),
            ("Product"  , 0, None, True, True, "g/L", None),
            ("Volume"   , 0, None, True, True, "L",   None),
            ("Feed"     , 0, None, True, True, "μL",  None),
            # fmt: on
        ]
        return (
            DataFrame(data, columns=list(self.table_schemas["timeseries_description"]))
            .astype(self.table_schemas["timeseries_description"])
            .set_index("variable")
        )

    def clean_table(self) -> DataFrame:
        with ZipFile(self.rawdata_paths["in_silico.zip"]) as files:
            dfs = {}
            for fname in files.namelist():
                key = int(fname.split(".csv")[0])
                with files.open(fname) as file:
                    df = pd.read_csv(file, index_col=0, parse_dates=[0])
                    dfs[key] = df.rename_axis(index="time")

        # Set index, dtype and sort.
        ts = (
            pd.concat(dfs, names=["run_id"])
            .reset_index()
            .set_index(["run_id", "time"])
            .sort_index()
            .astype("float32[pyarrow]")
        )
        ts = remove_outliers(ts, self._timeseries_description())
        return ts

    def download_file(self, fname: str) -> None:
        r"""Download the dataset."""
        self.LOGGER.info("Copying data files into %s.", self.rawdata_paths[fname])
        with resources.path(examples, fname) as path:
            shutil.copy(path, self.rawdata_paths[fname])
