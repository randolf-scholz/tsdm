r"""MIMIC-III Clinical Database.

Abstract
--------
MIMIC-III is a large, freely available database comprising de-identified health-related
data associated with over forty thousand patients who stayed in critical care units of
the Beth Israel Deaconess Medical Center between 2001 and 2012.
The database includes information such as demographics, vital sign measurements made at
the bedside (~1 data point per hour), laboratory test results, procedures, medications,
caregiver notes, imaging reports, and mortality (including post-hospital discharge).

MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical
decision-rule improvement, and electronic tool development. It is notable for three
factors: it is freely available to researchers worldwide; it encompasses a diverse and
very large population of ICU patients; and it contains highly granular data, including
vital signs, laboratory results, and medications.
"""

__all__ = [
    "RAWDATA_SCHEMA",
    "STATIC_COVARIATES_SCHEMA",
    "TIMESERIES_SCHEMA",
    "MIMIC_III_DeBrouwer2019",
]


from typing import Literal

import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import validate_schema

RAWDATA_SCHEMA = {
    "UNIQUE_ID": pl.Int16,
    "TIME_STAMP": pl.Int16,
    "LABEL_CODE": pl.Int16,
    "VALUENUM": pl.Float32,
    "MEAN": pl.Float32,
    "STD": pl.Float32,
    "VALUENORM": pl.Float32,
}
TIMESERIES_SCHEMA = {
    "UNIQUE_ID": pl.Int16,
    "TIME_STAMP": pl.Int16,
    **{str(label): pl.Float32 for label in range(96)},
}
STATIC_COVARIATES_SCHEMA = {
    "LABEL_CODE": pl.String,
    "count": pl.Float32,
    "mean": pl.Float32,
    "std": pl.Float32,
    "min": pl.Float32,
    "25%": pl.Float32,
    "50%": pl.Float32,
    "75%": pl.Float32,
    "max": pl.Float32,
}

type Key = Literal["timeseries", "static_covariates"]


class MIMIC_III_DeBrouwer2019(DatasetBase[Key, pl.DataFrame]):
    r"""MIMIC-III Clinical Database.

    MIMIC-III is a large, freely-available database comprising de-identified health-related data
    associated with over forty thousand patients who stayed in critical care units of the Beth
    Israel Deaconess Medical Center between 2001 and 2012. The database includes information such
    as demographics, vital sign measurements made at the bedside (~1 data point per hour),
    laboratory test results, procedures, medications, caregiver notes, imaging reports, and
    mortality (including post-hospital discharge).

    MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical decision-rule
    improvement, and electronic tool development. It is notable for three factors: it is freely
    available to researchers worldwide; it encompasses a diverse and very large population of ICU
    patients; and it contains highly granular data, including vital signs, laboratory results,
    and medications.

    Notes:
        NOTE: ``TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))``
        and ``bin_k = 10``
        i.e. ``TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)``
        i.e. ``TIME_STAMP ≈ 10*total_hours``
        so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    SOURCE_URL = r"https://physionet.org/content/mimiciii/get-zip/1.4/"
    INFO_URL = r"https://physionet.org/content/mimiciii/1.4/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/edebrouwer/gru_ode_bayes/"

    table_names = ["timeseries", "static_covariates"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["complete_tensor.csv"]
    rawdata_hashes = {
        "complete_tensor.csv": "sha256:8e884a916d28fd546b898b54e20055d4ad18d9a7abe262e15137080e9feb4fc2",
    }
    rawdata_shapes = {"complete_tensor.csv": (3_082_224, 7)}
    rawdata_schemas = {"complete_tensor.csv": RAWDATA_SCHEMA}
    table_schemas = {
        "timeseries": TIMESERIES_SCHEMA,
        "static_covariates": STATIC_COVARIATES_SCHEMA,
    }
    table_shapes = {
        "timeseries": (552_327, 98),
        "static_covariates": (96, 9),
    }

    timeseries: pl.DataFrame
    static_covariates: pl.DataFrame

    def clean_static_covariates(self) -> pl.DataFrame:
        r"""Create per-channel summary statistics."""
        target_schema = self.table_schemas["static_covariates"]
        statistics = list(target_schema)[1:]
        table = self.timeseries.drop("UNIQUE_ID", "TIME_STAMP")

        return (
            table.describe(
                percentiles=[0.25, 0.5, 0.75],
                interpolation="linear",
            )
            .filter(pl.col("statistic").is_in(statistics))
            .unpivot(
                index="statistic",
                variable_name="LABEL_CODE",
                value_name="value",
            )
            .pivot(
                on="statistic",
                index="LABEL_CODE",
                values="value",
                aggregate_function="first",
            )
            .select(
                pl.col(column).cast(dtype) for column, dtype in target_schema.items()
            )
            .sort(pl.col("LABEL_CODE").cast(pl.Int16))
        )

    def clean_timeseries(self) -> pl.DataFrame:
        self.LOGGER.info("Loading main file.")
        rawdata_schema = self.rawdata_schemas["complete_tensor.csv"]
        target_schema = self.table_schemas["timeseries"]
        rawdata_path = self.rawdata_paths["complete_tensor.csv"]
        validate_schema(rawdata_path, rawdata_schema)
        table = pl.read_csv(
            rawdata_path,
            schema=rawdata_schema,
        ).select(*rawdata_schema)

        # Check shape.
        if table.shape != self.rawdata_shapes["complete_tensor.csv"]:
            raise ValueError(
                f"The {table.shape=} is not correct. Please apply the modified"
                " preprocessing using bin_k=2, as outlined inthe appendix. The"
                " resulting tensor should have 3082224 rows and 7 columns."
            )

        # Extract Original Data Table.
        return (
            table.select("UNIQUE_ID", "TIME_STAMP", "LABEL_CODE", "VALUENUM")
            .pivot(
                on="LABEL_CODE",
                index=["UNIQUE_ID", "TIME_STAMP"],
                values="VALUENUM",
                aggregate_function="first",
            )
            .select(*target_schema)
            .sort("UNIQUE_ID", "TIME_STAMP")
        )

    def load_table(self, key: Key, /) -> pl.DataFrame:
        r"""Load a cleaned table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])

    def get_rawdata_file(self, _: str, /) -> None:
        if not self.rawdata_files_exist():
            raise RuntimeError(
                "Please manually apply the preprocessing code found at"
                f" {self.GITHUB_URL}.\nPut the resulting file 'complete_tensor.csv' in"
                f" {self.RAWDATA_DIR}.\nThe cleaning code is not included in this"
                " package because the original.\nauthors did not provide a license"
                " for it."
            )

    # FIXME: https://github.com/numpy/numpy/issues/24738
    def make_histograms(self) -> tuple[Figure, NDArray]:
        r"""Make histograms of the timeseries."""
        fig, axes = plt.subplots(
            16,
            6,
            figsize=(20, 32),
            constrained_layout=True,
            sharey=True,
            squeeze=False,
        )

        ts = self.timeseries.drop("UNIQUE_ID", "TIME_STAMP")
        for column, ax in zip(ts.columns, axes.flatten(), strict=True):
            ax.hist(
                ts.get_column(column).drop_nulls().to_numpy(),
                density=True,
                log=True,
                bins=20,
            )
            ax.set_ylim(10**-6, 1)

        return fig, axes
