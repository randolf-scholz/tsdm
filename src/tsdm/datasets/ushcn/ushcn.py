r"""UNITED STATES HISTORICAL CLIMATOLOGY NETWORK (USHCN) Daily Dataset."""

__all__ = [
    # Constants
    "STATIC_COVARIATES_METADATA",
    "STATE_CODES",
    "TIMESERIES_METADATA",
    # Classes
    "USHCN",
]

import tarfile
import warnings
from typing import Literal

import pandas as pd
from pandas import DataFrame

from tsdm.datasets.base import DatasetBase
from tsdm.datasets.schemas import DEFAULT_METADATA_SCHEMA
from tsdm.datatools import InlineTable, make_dataframe, remove_outliers

TIMESERIES_METADATA: InlineTable = {
    "data": [
        ("PRCP", "float32[pyarrow]",    0, None,  True,  True, "0.01 in", "precipitation"      ),
        ("SNOW", "float32[pyarrow]",    0, None,  True,  True, "0.1 in" , "snowfall"           ),
        ("SNWD", "float32[pyarrow]",    0, None,  True,  True, "in"     , "snow depth"         ),
        ("TMAX", "float32[pyarrow]", -100,  150, False, False, "℉"     , "maximum temperature"),
        ("TMIN", "float32[pyarrow]", -100,  150, False, False, "℉"     , "minimum temperature"),
    ],
    "schema": DEFAULT_METADATA_SCHEMA,
    "index": ["variable"],
}  # fmt: skip

STATIC_COVARIATES_METADATA: InlineTable = {
    "data": [
        ("LATITUDE"   , "float32[pyarrow]",  -90,    90, True, True,  "°" , "latitude"   ),
        ("LONGITUDE"  , "float32[pyarrow]", -180,   180, True, True,  "°" , "longitude"  ),
        ("ELEVATION"  , "float32[pyarrow]", -100, 10000, True, True,  "m" , "elevation"  ),
        ("STATE"      , "string[pyarrow]" , None,  None, True, True,  None, "state"      ),
        ("NAME"       , "string[pyarrow]" , None,  None, True, True,  None, "name"       ),
        ("COMPONENT_1", "int32[pyarrow]"  ,    0,  None, True, True,  None, "station ID" ),
        ("COMPONENT_2", "int32[pyarrow]"  ,    0,  None, True, True,  None, "station ID" ),
        ("COMPONENT_3", "int32[pyarrow]"  ,    0,  None, True, True,  None, "station ID" ),
        ("UTC_OFFSET" , "string[pyarrow]" , None,  None, True, True,  "h" ,  "UTC offset"),
    ],
    "schema": DEFAULT_METADATA_SCHEMA,
    "index": ["variable"],
}  # fmt: skip

STATE_CODES: InlineTable = {
    "data": [
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
    ],
    "schema": {
        "ID": "string[pyarrow]",
        "Abbr.": "string[pyarrow]",
        "State": "string[pyarrow]",
    },
}

type Key = Literal[
    "timeseries",
    "timeseries_metadata",
    "static_covariates",
    "static_covariates_metadata",
    "state_codes",
    "raw_timeseries",
]


class USHCN(DatasetBase[Key, DataFrame]):
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
        "pub12/ushcn_daily/data_format.txt"        : \
            "sha256:0fecc3670ea4c00d28385b664a9320d45169dbaea6d7ea962b41274ae77b07ca",
        "pub12/ushcn_daily/ushcn-stations.txt"     : \
            "sha256:002a25791b8c48dd39aa63e438c33a4f398b57cfa8bac28e0cde911d0c10e024",
        "pub12/ushcn_daily/station_file_format.txt": \
            "sha256:4acc15ec28aed24f25b75405f611bd719c5f36d6a05c36392d95f5b08a3b798b",
        "pub12/ushcn_daily/us.txt.gz"              : \
            "sha256:4cc2223f92e4c8e3bcb00bd4b13528c017594a2385847a611b96ec94be3b8192",
    }  # fmt: skip
    rawdata_schemas = {
        "timeseries": {
            "COOP_ID" : "int32[pyarrow]",
            "YEAR"    : "int16[pyarrow]",
            "MONTH"   : "int8[pyarrow]",
            "ELEMENT" : "string[pyarrow]",
            "VALUE"   : "int16[pyarrow]",
            "MFLAG"   : "string[pyarrow]",
            "QFLAG"   : "string[pyarrow]",
            "SFLAG"   : "string[pyarrow]",
        },
        "static_covariates": {
            "COOP_ID"     : "int32[pyarrow]",
            "LATITUDE"    : "float32[pyarrow]",
            "LONGITUDE"   : "float32[pyarrow]",
            "ELEVATION"   : "float32[pyarrow]",
            "STATE"       : "string[pyarrow]",
            "NAME"        : "string[pyarrow]",
            "COMPONENT_1" : "int32[pyarrow]",
            "COMPONENT_2" : "int32[pyarrow]",
            "COMPONENT_3" : "int32[pyarrow]",
            "UTC_OFFSET"  : "string[pyarrow]",
        },
    }  # fmt: skip

    table_shapes = {
        "timeseries"                 : (44497877, 5),
        "timeseries_metadata"        : (5, 7),
        "static_covariates"          : (1218, 9),
        "static_covariates_metadata" : (9, 7),
        "state_codes"                : (48, 3),
    }  # fmt: skip

    table_schemas = {
        "timeseries": {
            "PRCP" : "int16[pyarrow]",
            "SNOW" : "int16[pyarrow]",
            "SNWD" : "int16[pyarrow]",
            "TMAX" : "int16[pyarrow]",
            "TMIN" : "int16[pyarrow]",
        },
        "static_covariates": {
            "LATITUDE"    : "float[pyarrow]",
            "LONGITUDE"   : "float[pyarrow]",
            "ELEVATION"   : "float[pyarrow]",
            "STATE"       : "string[pyarrow]",
            "NAME"        : "string[pyarrow]",
            "COMPONENT_1" : "int32[pyarrow]",
            "COMPONENT_2" : "int32[pyarrow]",
            "COMPONENT_3" : "int32[pyarrow]",
            "UTC_OFFSET"  : "int8[pyarrow]",
        },
        "timeseries_metadata": DEFAULT_METADATA_SCHEMA,
        "static_covariates_metadata": DEFAULT_METADATA_SCHEMA,
    }  # fmt: skip

    def clean_table(self, key: Key = "timeseries") -> DataFrame:
        match key:
            case "timeseries":
                return self._clean_timeseries()
            case "raw_timeseries":
                return self._clean_raw_timeseries()
            case "static_covariates":
                return self._clean_static_covariates()
            case "state_codes":
                return make_dataframe(**STATE_CODES)
            case "timeseries_metadata":
                return make_dataframe(**TIMESERIES_METADATA)
            case "static_covariates_metadata":
                return make_dataframe(**STATIC_COVARIATES_METADATA)
            case _:
                raise KeyError(f"Unknown key: {key}")

    def _clean_static_covariates(self) -> DataFrame:
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

        # pandas wants list[tuple[int, int]], 0 indexed, half open intervals.
        stations_cspecs = [(a - 1, b) for a, b in stations_colspecs.values()]

        na_values = {
            "ELEVATION": ["-999.9"],
            "COMPONENT_1": ["------"],
            "COMPONENT_2": ["------"],
            "COMPONENT_3": ["------"],
        }

        with tarfile.open(rawdata_path, "r:gz") as archive:
            member = archive.extractfile("pub12/ushcn_daily/ushcn-stations.txt")
            if member is None:
                raise FileNotFoundError
            with member as file:
                self.LOGGER.info("Loading stations file...")
                static_covariates = pd.read_fwf(
                    file,
                    colspecs=stations_cspecs,
                    dtype=rawdata_schema,
                    names=stations_colspecs,
                    na_values=na_values,
                    dtype_backend="pyarrow",
                ).set_index("COOP_ID")

        self.LOGGER.info("Removing outliers from static_covariates.")
        static_covariates = remove_outliers(static_covariates, metadata)

        self.LOGGER.info("Dropping completely missing rows.")
        static_covariates = static_covariates.dropna(how="all", axis="index")

        static_covariates["UTC_OFFSET"] = (
            static_covariates["UTC_OFFSET"]
            .str.strip()
            .str.removeprefix("+")
            .replace({"": pd.NA})
            .astype("int8[pyarrow]")
        )

        # ensure table_schema is met
        if missing_cols := (target_schema.keys() - set(static_covariates.columns)):
            raise ValueError(f"Missing columns in static_covariates: {missing_cols}")

        return static_covariates.reindex(columns=target_schema).astype(target_schema)

    def _clean_timeseries(self) -> DataFrame:
        self.LOGGER.info("Creating simplified timeseries table.")
        table = self.tables["raw_timeseries"]
        target_schema = self.table_schemas["timeseries"]

        self.LOGGER.info("dropping all data with raised quality flags.")
        table = table.loc[table["QFLAG"].isna()]

        # convert from tall to wide with columns PRCP, SNOW, SNWD, TMAX, TMIN
        self.LOGGER.info("Performing pivot operation.")
        ts = table.pivot(columns="ELEMENT", values="VALUE")

        self.LOGGER.info("Removing outliers from timeseries.")
        ts = remove_outliers(ts, self.timeseries_metadata)

        self.LOGGER.info("Dropping completely missing rows.")
        ts = ts.dropna(how="all", axis="index")

        # ensure table_schema is met
        if missing_cols := (target_schema.keys() - set(ts.columns)):
            raise ValueError(f"Missing columns in timeseries: {missing_cols}")

        return ts.reindex(columns=target_schema).astype(target_schema)

    def _clean_raw_timeseries(self) -> DataFrame:
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
        rawdata_path = self.rawdata_paths["ushcn_daily.tar.gz"]
        rawdata_schema = self.rawdata_schemas["timeseries"]

        # column schema: (start, stop)
        colspecs: dict[str | tuple[str, int], tuple[int, int]] = {
            "COOP_ID" : (1, 6),
            "YEAR"    : (7, 10),
            "MONTH"   : (11, 12),
            "ELEMENT" : (13, 16),
        }  # fmt: skip

        # Add columns for each day of the month.
        for k, i in enumerate(range(17, 258, 8)):
            colspecs |= {
                ("VALUE", k + 1): (i, i + 4),
                ("MFLAG", k + 1): (i + 5, i + 5),
                ("QFLAG", k + 1): (i + 6, i + 6),
                ("SFLAG", k + 1): (i + 7, i + 7),
            }

        # pandas wants list[tuple[int, int]], 0 indexed, half open intervals.
        cspec = [(a - 1, b) for a, b in colspecs.values()]

        MFLAGS_DTYPE = pd.CategoricalDtype(list("BDHKLOPTW"))
        QFLAGS_DTYPE = pd.CategoricalDtype(list("DGIKLMNORSTWXZ"))
        SFLAGS_DTYPE = pd.CategoricalDtype(list("067ABFGHKMNRSTUWXZ"))
        ELEMENTS_DTYPE = pd.CategoricalDtype(["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"])
        VALUES_DTYPE = "int16[pyarrow]"

        updated_dtypes = {
            "COOP_ID" : "int32[pyarrow]",
            "YEAR"    : "int16[pyarrow]",
            "MONTH"   : "int8[pyarrow]",
            "ELEMENT" : ELEMENTS_DTYPE,
            "VALUE"   : VALUES_DTYPE,
            "MFLAG"   : MFLAGS_DTYPE,
            "QFLAG"   : QFLAGS_DTYPE,
            "SFLAG"   : SFLAGS_DTYPE,
        }  # fmt: skip

        # dtypes but with the same index as colspec.
        column_dtypes = {
            key: rawdata_schema[key[0]]
            if isinstance(key, tuple)
            else rawdata_schema[key]
            for key in colspecs
        }

        # per column values to be interpreted as nan
        na_values = {("VALUE", k): ["-9999"] for k in range(1, 32)}

        self.LOGGER.info("Loading main file...")
        with tarfile.open(rawdata_path, "r:gz") as archive:
            member = archive.extractfile("pub12/ushcn_daily/us.txt.gz")
            if member is None:
                raise FileNotFoundError
            with member as file:
                ds = pd.read_fwf(
                    file,
                    colspecs=cspec,
                    names=colspecs,
                    na_values=na_values,
                    dtype=column_dtypes,
                    compression="gzip",
                ).rename_axis(index="ID")

        self.LOGGER.info("Splitting dataframe...")
        # convert data part (VALUES, SFLAGS, MFLAGS, QFLAGS) to stand-alone dataframe
        id_cols = ["COOP_ID", "YEAR", "MONTH", "ELEMENT"]
        data_cols = [col for col in ds.columns if col not in id_cols]
        data, index = ds[data_cols], ds[id_cols]
        del ds

        self.LOGGER.info("Cleaning up columns...")
        # Turn tuple[VALUE/FLAG, DAY] indices to multi-index:
        data.columns = pd.MultiIndex.from_frame(
            DataFrame(data_cols, columns=["VAR", "DAY"]).astype(
                {
                    "VAR": "string[pyarrow]",
                    "DAY": "int8[pyarrow]",
                }
            )
        )

        self.LOGGER.info("Stacking on FLAGS and VALUES columns...")
        # stack on day, this will collapse (VALUE1, ..., VALUE31) into a single VALUE column.
        data = (
            data.stack(level="DAY")
            .reset_index(level="DAY")
            .astype(
                {  # correct dtypes after stacking operation
                    "DAY": "int8[pyarrow]",
                    "VALUE": VALUES_DTYPE,
                    "MFLAG": MFLAGS_DTYPE,
                    "QFLAG": QFLAGS_DTYPE,
                    "SFLAG": SFLAGS_DTYPE,
                }
            )
        )

        self.LOGGER.info("Merging on ID columns...")
        data = data.join(index, how="inner").astype(updated_dtypes)

        self.LOGGER.info("Creating time index...")
        date_cols = ["YEAR", "MONTH", "DAY"]
        dates = pd.to_datetime(data[date_cols], errors="coerce").astype(
            "date32[pyarrow]"
        )
        data = (
            data.assign(DATE=dates)
            .drop(columns=date_cols)
            .dropna(subset=["DATE", "VALUE"])
        )

        self.LOGGER.info("Set index and sort...")
        data = (
            data.set_index(["COOP_ID", "DATE"])
            .reindex(columns=["ELEMENT", "MFLAG", "QFLAG", "SFLAG", "VALUE"])
            .sort_values(by=["COOP_ID", "DATE", "ELEMENT"])
        )

        return data
