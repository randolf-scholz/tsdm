r"""UNITED STATES HISTORICAL CLIMATOLOGY NETWORK (USHCN) Daily Dataset."""

__all__ = [
    # Constants
    "STATIC_COVARIATES_METADATA",
    "STATE_CODES",
    "TIMESERIES_METADATA",
    "STATE_CODES_SCHEMA",
    "TIMESERIES_SCHEMA",
    "RAWDATA_STATIC_COVARIATES_SCHEMA",
    "RAWDATA_TIMESERIES_SCHEMA",
    "STATIC_COVARIATES_SCHEMA",
    "RAWTIMESERIES_SCHEMA",
    "METADATA_SCHEMA",
    # Classes
    "USHCN",
]

import gzip
import tarfile
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal, Optional

import polars as pl

from tsdm.datasets.base import PolarsDataset
from tsdm.datatools import remove_outliers
from tsdm.types.aliases import FilePath, FileStream

METADATA_SCHEMA = {
    "variable"        : pl.String,
    "dtype"           : pl.String,
    "lower_bound"     : pl.Float64,
    "upper_bound"     : pl.Float64,
    "lower_inclusive" : pl.Boolean,
    "upper_inclusive" : pl.Boolean,
    "unit"            : pl.String,
    "description"     : pl.String,
}  # fmt: skip

TIMESERIES_METADATA = [
    ("COOP_ID", "Int32"  , None, None, None, None,      None, "station identifier"),
    ("DATE"   , "Date"   , None, None, None, None,      None, "observation date"  ),
    ("PRCP", "Float32",    0, None,  True,  True, "0.01 in", "precipitation"      ),
    ("SNOW", "Float32",    0, None,  True,  True, "0.1 in" , "snowfall"           ),
    ("SNWD", "Float32",    0, None,  True,  True, "in"     , "snow depth"         ),
    ("TMAX", "Float32", -100,  150, False, False, "℉"      , "maximum temperature"),
    ("TMIN", "Float32", -100,  150, False, False, "℉"      , "minimum temperature"),
]  # fmt: skip

STATIC_COVARIATES_METADATA = [
    ("COOP_ID"   , "Int32"  , None, None, None, None,  None, "station identifier"),
    ("LATITUDE"   , "Float32",  -90,    90, True, True,  "°" , "latitude"   ),
    ("LONGITUDE"  , "Float32", -180,   180, True, True,  "°" , "longitude"  ),
    ("ELEVATION"  , "Float32", -100, 10000, True, True,  "m" , "elevation"  ),
    ("STATE"      , "String" , None,  None, True, True,  None, "state"      ),
    ("NAME"       , "String" , None,  None, True, True,  None, "name"       ),
    ("COMPONENT_1", "Int32"  ,    0,  None, True, True,  None, "station ID" ),
    ("COMPONENT_2", "Int32"  ,    0,  None, True, True,  None, "station ID" ),
    ("COMPONENT_3", "Int32"  ,    0,  None, True, True,  None, "station ID" ),
    ("UTC_OFFSET" , "Int8"   , None,  None, True, True,  "h" , "UTC offset" ),
]  # fmt: skip

STATE_CODES = [
    ("01", "AL", "Alabama"),
    ("02", "AZ", "Arizona"),
    ("03", "AR", "Arkansas"),
    ("04", "CA", "California"),
    ("05", "CO", "Colorado"),
    ("06", "CT", "Connecticut"),
    ("07", "DE", "Delaware"),
    ("08", "FL", "Florida"),
    ("09", "GA", "Georgia"),
    ("10", "ID", "Idaho"),
    ("11", "IL", "Idaho"),
    ("12", "IN", "Indiana"),
    ("13", "IA", "Iowa"),
    ("14", "KS", "Kansas"),
    ("15", "KY", "Kentucky"),
    ("16", "LA", "Louisiana"),
    ("17", "ME", "Maine"),
    ("18", "MD", "Maryland"),
    ("19", "MA", "Massachusetts"),
    ("20", "MI", "Michigan"),
    ("21", "MN", "Minnesota"),
    ("22", "MS", "Mississippi"),
    ("23", "MO", "Missouri"),
    ("24", "MT", "Montana"),
    ("25", "NE", "Nebraska"),
    ("26", "NV", "Nevada"),
    ("27", "NH", "NewHampshire"),
    ("28", "NJ", "NewJersey"),
    ("29", "NM", "NewMexico"),
    ("30", "NY", "NewYork"),
    ("31", "NC", "NorthCarolina"),
    ("32", "ND", "NorthDakota"),
    ("33", "OH", "Ohio"),
    ("34", "OK", "Oklahoma"),
    ("35", "OR", "Oregon"),
    ("36", "PA", "Pennsylvania"),
    ("37", "RI", "RhodeIsland"),
    ("38", "SC", "SouthCarolina"),
    ("39", "SD", "SouthDakota"),
    ("40", "TN", "Tennessee"),
    ("41", "TX", "Texas"),
    ("42", "UT", "Utah"),
    ("43", "VT", "Vermont"),
    ("44", "VA", "Virginia"),
    ("45", "WA", "Washington"),
    ("46", "WV", "WestVirginia"),
    ("47", "WI", "Wisconsin"),
    ("48", "WY", "Wyoming"),
]
STATE_CODES_SCHEMA = {"ID": pl.String, "Abbr.": pl.String, "State": pl.String}

RAWDATA_TIMESERIES_SCHEMA = {
    "COOP_ID": pl.Int32,
    "YEAR": pl.Int16,
    "MONTH": pl.Int8,
    "ELEMENT": pl.Categorical,
    "VALUE": pl.Int16,
    "MFLAG": pl.Categorical,
    "QFLAG": pl.Categorical,
    "SFLAG": pl.Categorical,
}
RAWDATA_STATIC_COVARIATES_SCHEMA = {
    "COOP_ID": pl.Int32,
    "LATITUDE": pl.Float32,
    "LONGITUDE": pl.Float32,
    "ELEVATION": pl.Float32,
    "STATE": pl.String,
    "NAME": pl.String,
    "COMPONENT_1": pl.Int32,
    "COMPONENT_2": pl.Int32,
    "COMPONENT_3": pl.Int32,
    "UTC_OFFSET": pl.String,
}
RAWTIMESERIES_SCHEMA = {
    "COOP_ID": pl.Int32,
    "DATE": pl.Date,
    "ELEMENT": pl.Categorical,
    "MFLAG": pl.Categorical,
    "QFLAG": pl.Categorical,
    "SFLAG": pl.Categorical,
    "VALUE": pl.Int16,
}
TIMESERIES_SCHEMA = {
    "COOP_ID": pl.Int32,
    "DATE": pl.Date,
    "PRCP": pl.Int16,
    "SNOW": pl.Int16,
    "SNWD": pl.Int16,
    "TMAX": pl.Int16,
    "TMIN": pl.Int16,
}
STATIC_COVARIATES_SCHEMA = {
    "COOP_ID": pl.Int32,
    "LATITUDE": pl.Float32,
    "LONGITUDE": pl.Float32,
    "ELEVATION": pl.Float32,
    "STATE": pl.String,
    "NAME": pl.String,
    "COMPONENT_1": pl.Int32,
    "COMPONENT_2": pl.Int32,
    "COMPONENT_3": pl.Int32,
    "UTC_OFFSET": pl.Int8,
}


def _read_fwf(
    source: FilePath | FileStream,
    /,
    *,
    colspecs: Mapping[str, tuple[int, int]],
    schema: Mapping[str, pl.DataType],
    null_values: Optional[Mapping[str, Sequence[str]]] = None,
) -> pl.LazyFrame:
    r"""Read a fixed-width file by slicing each string row.

    The column specifications are one-indexed, closed intervals, matching the
    USHCN documentation.

    TODO: Replace this when Polars provides a native ``read_fwf`` implementation.
    """
    # FIXME: https://github.com/pola-rs/polars/issues/3151
    # FIXME: https://github.com/apache/arrow/issues/33404
    warnings.warn(
        "This can take a while to run, "
        "refactor if read_fwf becomes available in polars or pyarrow."
        "\nSee:"
        "\n - https://github.com/pola-rs/polars/issues/3151"
        "\n - https://github.com/apache/arrow/issues/33404",
        stacklevel=2,
    )
    rows = pl.scan_lines(source, name="line")  # type: ignore

    nulls = null_values or {}
    columns = []
    for name, (start, stop) in colspecs.items():
        field = pl.col("line").str.slice(start - 1, stop - start + 1).str.strip_chars()
        missing = field.eq("")
        if values := nulls.get(name):
            missing |= field.is_in(values)
        columns.append(
            pl.when(missing)
            .then(None)
            .otherwise(field)
            .cast(schema[name], strict=True)
            .alias(name)
        )
    return rows.select(columns)


type Key = Literal[
    "timeseries",
    "timeseries_metadata",
    "static_covariates",
    "static_covariates_metadata",
    "state_codes",
    "raw_timeseries",
]


class USHCN(PolarsDataset[Key]):
    r"""UNITED STATES HISTORICAL CLIMATOLOGY NETWORK (USHCN) Daily Dataset.

    U.S. Historical Climatology Network (USHCN) data are used to quantify national and
    regional-scale temperature changes in the contiguous United States (CONUS).
    The dataset provides adjustments for systematic, non-climatic changes that bias
    temperature trends of monthly temperature records of long-term COOP stations.
    USHCN is a designated subset of the NOAA Cooperative Observer Program (COOP)
    Network, with sites selected according to their spatial coverage, record length,
    data completeness, and historical stability.

    .. rubric:: Stations Data

    +----------+---------+-----------+
    | Variable | Columns | Type      |
    +==========+=========+===========+
    | COOP ID  | 1-6     | Character |
    +----------+---------+-----------+
    | YEAR     | 7-10    | Integer   |
    +----------+---------+-----------+
    | MONTH    | 11-12   | Integer   |
    +----------+---------+-----------+
    | ELEMENT  | 13-16   | Character |
    +----------+---------+-----------+
    | VALUE1   | 17-21   | Integer   |
    +----------+---------+-----------+
    | MFLAG1   | 22      | Character |
    +----------+---------+-----------+
    | QFLAG1   | 23      | Character |
    +----------+---------+-----------+
    | SFLAG1   | 24      | Character |
    +----------+---------+-----------+
    |     ⋮    |    ⋮    |     ⋮     |
    +----------+---------+-----------+
    | VALUE31  | 257-261 | Integer   |
    +----------+---------+-----------+
    | MFLAG31  | 262     | Character |
    +----------+---------+-----------+
    | QFLAG31  | 263     | Character |
    +----------+---------+-----------+
    | SFLAG31  | 264     | Character |
    +----------+---------+-----------+

    .. rubric: Station Variables

    - COOP ID	is the U.S. Cooperative Observer Network station identification code.
      Note that the first two digits in the Coop Id correspond to the state.
    - YEAR		is the year of the record.
    - MONTH	is the month of the record.
    - ELEMENT	is the element type. There are five possible values
        - PRCP = precipitation (hundredths of inches)
        - SNOW = snowfall (tenths of inches)
        - SNWD = snow depth (inches)
        - TMAX = maximum temperature (degrees F)
        - TMIN = minimum temperature (degrees F)
    - VALUE1	is the value on the first day of the month (missing = -9999).
    - MFLAG1	is the measurement flag for the first day of the month. There are five possible values:
        - Blank = no measurement information applicable
        - B = precipitation total formed from two 12-hour totals
        - D = precipitation total formed from four six-hour totals
        - H = represents highest or lowest hourly temperature
        - L = temperature appears to be lagged with respect to reported hour of observation
        - P = identified as "missing presumed zero" in DSI 3200 and 3206
        - T = trace of precipitation, snowfall, or snow depth
    - QFLAG1	is the quality flag for the first day of the month. There are fourteen possible values:
        - Blank = did not fail any quality assurance check
        - D = failed duplicate check
        - G = failed gap check
        - I = failed internal consistency check
        - K = failed streak/frequent-value check
        - L = failed check on length of multiday period
        - M = failed megaconsistency check
        - N = failed naught check
        - O = failed climatological outlier check
        - R = failed lagged range check
        - S = failed spatial consistency check
        - T = failed temporal consistency check
        - W = temperature too warm for snow
        - X = failed bounds check
        - Z = flagged as a result of an official Datzilla investigation
    - SFLAG1	is the source flag for the first day of the month. There are fifteen possible values:
        - Blank = No source (e.g., data value missing)
        - 0 = U.S. Cooperative Summary of the Day (NCDC DSI-3200)
        - 6 = CDMP Cooperative Summary of the Day (NCDC DSI-3206)
        - 7 = U.S. Cooperative Summary of the Day -- Transmitted via WxCoder3 (NCDC DSI-3207)
        - A = U.S. Automated Surface Observing System (ASOS) real-time data (since January 1, 2006)
        - B = U.S. ASOS data for October 2000-December 2005 (NCDC DSI-3211)
        - F = U.S. Fort Data
        - G = Official Global Climate Observing System (GCOS) or other government-supplied data
        - H = High Plains Regional Climate Center real-time data
        - M = Monthly METAR Extract (additional ASOS data)
        - N = Community Collaborative Rain, Hail, and Snow (CoCoRaHS)
        - R = NCDC Reference Network Database (Climate Reference Network and Historical Climatology Network-Modernized)
        - S = Global Summary of the Day (NCDC DSI-9618)

    .. rubric:: Stations Meta-Data

    +-------------+---------+-----------+
    | Variable    | Columns | Type      |
    +=============+=========+===========+
    | COOP ID     | 1-6     | Character |
    +-------------+---------+-----------+
    | LATITUDE    | 8-15    | Real      |
    +-------------+---------+-----------+
    | LONGITUDE   | 17-25   | Real      |
    +-------------+---------+-----------+
    | ELEVATION   | 27-32   | Real      |
    +-------------+---------+-----------+
    | STATE       | 34-35   | Character |
    +-------------+---------+-----------+
    | NAME        | 37-66   | Character |
    +-------------+---------+-----------+
    | COMPONENT 1 | 68-73   | Character |
    +-------------+---------+-----------+
    | COMPONENT 2 | 75-80   | Character |
    +-------------+---------+-----------+
    | COMPONENT 3 | 82-87   | Character |
    +-------------+---------+-----------+
    | UTC OFFSET  | 89-90   | Integer   |
    +-------------+---------+-----------+

    .. rubric:: Station Meta-Data Variables

    - COOP_ID		is the U.S. Cooperative Observer Network station identification code. Note that
      the first two digits in the Coop ID correspond to the assigned state number (see Table 1 below).
    - LATITUDE	is latitude of the station (in decimal degrees).
    - LONGITUDE	is the longitude of the station (in decimal degrees).
    - ELEVATION	is the elevation of the station (in meters, missing = -999.9).
    - STATE		is the U.S. postal code for the state.
    - NAME		is the name of the station location.
    - COMPONENT_1	is the Coop Id for the first station (in chronologic order) whose records were
      joined with those of the USHCN site to form a longer time series. "------" indicates "not applicable".
    - COMPONENT_2	is the Coop Id for the second station (if applicable) whose records were joined
      with those of the USHCN site to form a longer time series.
    - COMPONENT_3	is the Coop Id for the third station (if applicable) whose records were joined
      with those of the USHCN site to form a longer time series.
    - UTC_OFFSET	is the time difference between Coordinated Universal Time (UTC) and local standard time
      at the station (i.e., the number of hours that must be added to local standard time to match UTC).
    """

    # https://www.ncei.noaa.gov/products/land-based-station/us-historical-climatology-network
    # https://data.ess-dive.lbl.gov/view/doi:10.3334/CDIAC/CLI.NDP019
    # https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-7b1e0d7f2fc3c43-20180727T175547656
    # "https://cdiac.ess-dive.lbl.gov/ftp/ushcn_daily/"
    # "https://cdiac.ess-dive.lbl.gov/epubs/ndp/ushcn/daily_doc.html"
    SOURCE_URL = "https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-7b1e0d7f2fc3c43-20180727T175547656"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = "https://data.ess-dive.lbl.gov/view/doi:10.3334/CDIAC/CLI.NDP019"
    r"""HTTP address containing additional information about the dataset."""

    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "timeseries_metadata",
        "static_covariates",
        "static_covariates_metadata",
        # extra tables
        "raw_timeseries",
        "state_codes",
    ]
    rawdata_files = ["ushcn_daily.tar.gz"]
    rawdata_hashes = {
        "ushcn_daily.tar.gz" : "sha256:a03598657a3b72c20f8ffa323d7265435243d7988b02d2dbbaab746c2ccae25f",
    }  # fmt: skip
    rawdata_content_hashes = {
        "pub12/ushcn_daily/data_format.txt":
            "sha256:0fecc3670ea4c00d28385b664a9320d45169dbaea6d7ea962b41274ae77b07ca",
        "pub12/ushcn_daily/ushcn-stations.txt":
            "sha256:002a25791b8c48dd39aa63e438c33a4f398b57cfa8bac28e0cde911d0c10e024",
        "pub12/ushcn_daily/station_file_format.txt":
            "sha256:4acc15ec28aed24f25b75405f611bd719c5f36d6a05c36392d95f5b08a3b798b",
        "pub12/ushcn_daily/us.txt.gz":
            "sha256:4cc2223f92e4c8e3bcb00bd4b13528c017594a2385847a611b96ec94be3b8192",
    }  # fmt: skip
    rawdata_schemas = {
        "timeseries": RAWDATA_TIMESERIES_SCHEMA,
        "static_covariates": RAWDATA_STATIC_COVARIATES_SCHEMA,
    }

    table_shapes = {
        "timeseries"                 : (44497877, 7),
        "timeseries_metadata"        : (7, 8),
        "static_covariates"          : (1218, 10),
        "static_covariates_metadata" : (10, 8),
        "state_codes"                : (48, 3),
    }  # fmt: skip

    table_schemas = {
        "timeseries": TIMESERIES_SCHEMA,
        "timeseries_metadata": METADATA_SCHEMA,
        "static_covariates": STATIC_COVARIATES_SCHEMA,
        "static_covariates_metadata": METADATA_SCHEMA,
        "state_codes": STATE_CODES_SCHEMA,
        "raw_timeseries": RAWTIMESERIES_SCHEMA,
    }  # fmt: skip

    def clean_table(self, key: Key, /) -> pl.DataFrame:
        match key:
            case "timeseries":
                return self._clean_timeseries()
            case "raw_timeseries":
                return self._clean_raw_timeseries()
            case "static_covariates":
                return self._clean_static_covariates()
            case "state_codes":
                return pl.DataFrame(
                    STATE_CODES, schema=STATE_CODES_SCHEMA, orient="row"
                )
            case "timeseries_metadata":
                return pl.DataFrame(
                    TIMESERIES_METADATA, schema=METADATA_SCHEMA, orient="row"
                )
            case "static_covariates_metadata":
                return pl.DataFrame(
                    STATIC_COVARIATES_METADATA, schema=METADATA_SCHEMA, orient="row"
                )
            case _:
                raise KeyError(f"Unknown key: {key}")

    def _clean_static_covariates(self) -> pl.DataFrame:
        rawdata_path = self.rawdata_paths["ushcn_daily.tar.gz"]
        rawdata_schema = self.rawdata_schemas["static_covariates"]
        target_schema = self.table_schemas["static_covariates"]
        metadata = self.static_covariates_metadata

        stations_colspecs = {
            "COOP_ID":     (1, 6),
            "LATITUDE":    (8, 15),
            "LONGITUDE":   (17, 25),
            "ELEVATION":   (27, 32),
            "STATE":       (34, 35),
            "NAME":        (37, 66),
            "COMPONENT_1": (68, 73),
            "COMPONENT_2": (75, 80),
            "COMPONENT_3": (82, 87),
            "UTC_OFFSET":  (89, 90),
        }  # fmt: skip

        na_values = {
            "ELEVATION": ("-999.9",),
            "COMPONENT_1": ("------",),
            "COMPONENT_2": ("------",),
            "COMPONENT_3": ("------",),
        }

        with tarfile.open(rawdata_path, "r:gz") as archive:
            member = archive.extractfile("pub12/ushcn_daily/ushcn-stations.txt")
            if member is None:
                raise FileNotFoundError
            with member as file:
                self.LOGGER.info("Loading stations file...")
                static_covariates = _read_fwf(
                    file,
                    colspecs=stations_colspecs,
                    schema=rawdata_schema,
                    null_values=na_values,
                ).collect()

        self.LOGGER.info("Removing outliers from static_covariates.")
        covariate_columns = [column for column in target_schema if column != "COOP_ID"]
        covariates = remove_outliers(
            static_covariates.select(*covariate_columns), metadata, drop=False
        )

        self.LOGGER.info("Dropping completely missing rows.")
        static_covariates = (
            static_covariates.select("COOP_ID")
            .hstack(covariates)
            .with_columns(
                pl.col("UTC_OFFSET")
                .str.strip_chars()
                .str.strip_prefix("+")
                .cast(pl.Int8)
            )
            .filter(
                pl.any_horizontal(
                    *(pl.col(column).is_not_null() for column in covariate_columns)
                )
            )
        )

        if missing_cols := (target_schema.keys() - set(static_covariates.columns)):
            raise ValueError(f"Missing columns in static_covariates: {missing_cols}")

        return static_covariates.select(*target_schema).sort("COOP_ID")

    def _clean_timeseries(self) -> pl.DataFrame:
        self.LOGGER.info("Creating simplified timeseries table.")
        table = self.tables["raw_timeseries"]
        target_schema = self.table_schemas["timeseries"]

        self.LOGGER.info("dropping all data with raised quality flags.")
        table = table.filter(pl.col("QFLAG").is_null())

        # convert from tall to wide with columns PRCP, SNOW, SNWD, TMAX, TMIN
        self.LOGGER.info("Performing pivot operation.")
        ts = table.pivot(
            on="ELEMENT",
            index=["COOP_ID", "DATE"],
            values="VALUE",
            aggregate_function=None,
        )

        self.LOGGER.info("Removing outliers from timeseries.")
        index_columns = {"COOP_ID", "DATE"}
        value_columns = [
            column for column, *_ in TIMESERIES_METADATA if column not in index_columns
        ]
        values = remove_outliers(
            ts.select(*value_columns), self.timeseries_metadata, drop=False
        )

        self.LOGGER.info("Dropping completely missing rows.")
        ts = (
            ts.select("COOP_ID", "DATE")
            .hstack(values)
            .filter(
                pl.any_horizontal(
                    *(pl.col(column).is_not_null() for column in value_columns)
                )
            )
        )

        if missing_cols := (target_schema.keys() - set(ts.columns)):
            raise ValueError(f"Missing columns in timeseries: {missing_cols}")

        return ts.select(*target_schema).sort("COOP_ID", "DATE")

    def _clean_raw_timeseries(self) -> pl.DataFrame:
        rawdata_path = self.rawdata_paths["ushcn_daily.tar.gz"]
        rawdata_schema = self.rawdata_schemas["timeseries"]

        # column schema: (start, stop)
        colspecs: dict[str, tuple[int, int]] = {
            "COOP_ID" : (1, 6),
            "YEAR"    : (7, 10),
            "MONTH"   : (11, 12),
            "ELEMENT" : (13, 16),
        }  # fmt: skip

        days = range(1, 32)
        fields = {
            "VALUE": (0, 5),
            "MFLAG": (5, 1),
            "QFLAG": (6, 1),
            "SFLAG": (7, 1),
        }
        colspecs |= {
            f"{field}_{day:02d}": (start + offset, start + offset + width - 1)
            for day, start in enumerate(range(17, 258, 8), start=1)
            for field, (offset, width) in fields.items()
        }
        day_columns = {
            field: [f"{field}_{day:02d}" for day in days] for field in fields
        }
        column_dtypes = {
            **{
                column: rawdata_schema[column]
                for column in rawdata_schema
                if column in colspecs
            },
            **{
                column: rawdata_schema[field]
                for field, columns in day_columns.items()
                for column in columns
            },
        }
        na_values = dict.fromkeys(day_columns["VALUE"], ("-9999",))

        self.LOGGER.info("Loading main file...")
        with tarfile.open(rawdata_path, "r:gz") as archive:
            member = archive.extractfile("pub12/ushcn_daily/us.txt.gz")
            if member is None:
                raise FileNotFoundError
            with member as file, gzip.GzipFile(fileobj=file) as gzip_file:
                self.LOGGER.info("Expanding daily observations...")
                data = (
                    _read_fwf(
                        gzip_file,  # type: ignore
                        colspecs=colspecs,
                        schema=column_dtypes,
                        null_values=na_values,
                    )
                    .select(
                        "COOP_ID",
                        "YEAR",
                        "MONTH",
                        "ELEMENT",
                        pl.lit(list(days), dtype=pl.List(pl.Int8)).alias("DAY"),
                        *(
                            pl.concat_list(*columns).alias(field)
                            for field, columns in day_columns.items()
                        ),
                    )
                    .explode("DAY", *fields)
                    .with_columns(
                        pl.concat_str(["YEAR", "MONTH", "DAY"], separator="-")
                        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                        .alias("DATE")
                    )
                    .drop("YEAR", "MONTH", "DAY")
                    .drop_nulls(["DATE", "VALUE"])
                    .select(*RAWTIMESERIES_SCHEMA)
                    .sort("COOP_ID", "DATE", "ELEMENT")
                    .collect(engine="streaming")
                )

        self.LOGGER.info("Created raw timeseries table.")

        return data
