r"""UNITED STATES HISTORICAL CLIMATOLOGY NETWORK (USHCN) Daily Dataset."""

__all__ = ["USHCN"]

import warnings

import pandas as pd
from pandas import DataFrame
from typing_extensions import Literal, TypeAlias

from tsdm.data import InlineTable, make_dataframe, remove_outliers
from tsdm.datasets.base import MultiTableDataset

KEY: TypeAlias = Literal[
    "timeseries",
    "timeseries_description",
    "metadata",
    "metadata_description",
    "state_codes",
    "raw_timeseries",
]

TIMESERIES_DESCRIPTION: InlineTable = {
    "data": [
        # fmt: off
        ("PRCP", "float32[pyarrow]",    0, None,  True,  True, "10²in", "precipitation"      ),
        ("SNOW", "float32[pyarrow]",    0, None,  True,  True, "10¹in", "snowfall"           ),
        ("SNWD", "float32[pyarrow]",    0, None,  True,  True, "in"   , "snow depth"         ),
        ("TMAX", "float32[pyarrow]", -100,  150, False, False, "°F"   , "maximum temperature"),
        ("TMIN", "float32[pyarrow]", -100,  150, False, False, "°F"   , "minimum temperature"),
        # fmt: on
    ],
    "schema": {
        # fmt: off
        "variable"        : "string[pyarrow]",
        "dtype"           : "string[pyarrow]",
        "lower_bound"     : "float32[pyarrow]",
        "upper_bound"     : "float32[pyarrow]",
        "lower_inclusive" : "bool[pyarrow]",
        "upper_inclusive" : "bool[pyarrow]",
        "unit"            : "string[pyarrow]",
        "description"     : "string[pyarrow]",
        # fmt: on
    },
    "index": ["variable"],
}

METADATA_DESCRIPTION: InlineTable = {
    "data": [
        # fmt: off
        ("LATITUDE"   , "float32[pyarrow]",  -90,    90, True, True,  "°" , "latitude"   ),
        ("LONGITUDE"  , "float32[pyarrow]", -180,   180, True, True,  "°" , "longitude"  ),
        ("ELEVATION"  , "float32[pyarrow]", -100, 10000, True, True,  "m" , "elevation"  ),
        ("STATE"      , "string[pyarrow]" , None,  None, True, True,  None, "state"      ),
        ("NAME"       , "string[pyarrow]" , None,  None, True, True,  None, "name"       ),
        ("COMPONENT_1", "int32[pyarrow]"  ,    0,  None, True, True,  None, "station ID" ),
        ("COMPONENT_2", "int32[pyarrow]"  ,    0,  None, True, True,  None, "station ID" ),
        ("COMPONENT_3", "int32[pyarrow]"  ,    0,  None, True, True,  None, "station ID" ),
        ("UTC_OFFSET" , "string[pyarrow]" , None,  None, True, True,  "h" ,  "UTC offset"),
        # fmt: on
    ],
    "schema": {
        # fmt: off
        "variable"        : "string[pyarrow]",
        "dtype"           : "string[pyarrow]",
        "lower_bound"     : "float32[pyarrow]",
        "upper_bound"     : "float32[pyarrow]",
        "lower_inclusive" : "bool[pyarrow]",
        "upper_inclusive" : "bool[pyarrow]",
        "unit"            : "string[pyarrow]",
        "description"     : "string[pyarrow]",
        # fmt: on
    },
    "index": ["variable"],
}

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


class USHCN(MultiTableDataset[KEY, DataFrame]):
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

    BASE_URL = "https://cdiac.ess-dive.lbl.gov/ftp/ushcn_daily/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = "https://cdiac.ess-dive.lbl.gov/epubs/ndp/ushcn/daily_doc.html"
    r"""HTTP address containing additional information about the dataset."""

    table_names = [  # pyright: ignore
        "timeseries",
        "timeseries_description",
        "metadata",
        "metadata_description",
        # extra tables
        "raw_timeseries",
        "state_codes",
    ]
    rawdata_files = [
        "data_format.txt",
        "ushcn-stations.txt",
        "station_file_format.txt",
        "us.txt.gz",
    ]
    rawdata_hashes = {
        "data_format.txt": (
            "sha256:0fecc3670ea4c00d28385b664a9320d45169dbaea6d7ea962b41274ae77b07ca"
        ),
        "ushcn-stations.txt": (
            "sha256:002a25791b8c48dd39aa63e438c33a4f398b57cfa8bac28e0cde911d0c10e024"
        ),
        "station_file_format.txt": (
            "sha256:4acc15ec28aed24f25b75405f611bd719c5f36d6a05c36392d95f5b08a3b798b"
        ),
        "us.txt.gz": (
            "sha256:4cc2223f92e4c8e3bcb00bd4b13528c017594a2385847a611b96ec94be3b8192"
        ),
    }
    rawdata_schemas = {
        "timeseries": {
            # fmt: off
            "COOP_ID" : "int32[pyarrow]",
            "YEAR"    : "int16[pyarrow]",
            "MONTH"   : "int8[pyarrow]",
            "ELEMENT" : "string[pyarrow]",
            "VALUE"   : "int16[pyarrow]",
            "MFLAG"   : "string[pyarrow]",
            "QFLAG"   : "string[pyarrow]",
            "SFLAG"   : "string[pyarrow]",
            # fmt: on
        },
        "metadata": {
            # fmt: off
            "COOP_ID"     : "int32[pyarrow]",
            "LATITUDE"    : "float32[pyarrow]",
            "LONGITUDE"   : "float32[pyarrow]",
            "ELEVATION"   : "float32[pyarrow]",
            "STATE"       : "string[pyarrow]",
            "NAME"        : "string[pyarrow]",
            "COMPONENT_1" : "int32[pyarrow]",
            "COMPONENT_2" : "int32[pyarrow]",
            "COMPONENT_3" : "int32[pyarrow]",
            "UTC_OFFSET"  : "int8[pyarrow]",
            # fmt: on
        },
    }
    table_shapes = {  # pyright: ignore
        "timeseries": (204771562, 5),
        "metadata": (1218, 9),
        "state_codes": (48, 3),
        "metadata_description": (9, 6),
        "timeseries_description": (8, 6),
    }
    table_schemas = {  # pyright: ignore
        "timeseries": {
            # fmt: off
            "PRCP" : "int16[pyarrow]",
            "SNOW" : "int16[pyarrow]",
            "SNWD" : "int16[pyarrow]",
            "TMAX" : "int16[pyarrow]",
            "TMIN" : "int16[pyarrow]",
            # fmt: on
        },
        "metadata": {
            # fmt: off
            "LATITUDE"    : "float[pyarrow]",
            "LONGITUDE"   : "float[pyarrow]",
            "ELEVATION"   : "float[pyarrow]",
            "STATE"       : "string[pyarrow]",
            "NAME"        : "string[pyarrow]",
            "COMPONENT_1" : "int32[pyarrow]",
            "COMPONENT_2" : "int32[pyarrow]",
            "COMPONENT_3" : "int32[pyarrow]",
            "UTC_OFFSET"  : "int8[pyarrow]",
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
        "metadata_description": {
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

    def clean_table(self, key: KEY = "timeseries") -> DataFrame:
        match key:
            case "timeseries":
                return self._clean_timeseries()
            case "raw_timeseries":
                return self._clean_raw_timeseries()
            case "metadata":
                return self._clean_metadata()
            case "state_codes":
                return make_dataframe(**STATE_CODES)
            case "timeseries_description":
                return make_dataframe(**TIMESERIES_DESCRIPTION)
            case "metadata_description":
                return make_dataframe(**METADATA_DESCRIPTION)
            case _:
                raise KeyError(f"Unknown key: {key}")

    def _clean_metadata(self) -> DataFrame:
        stations_colspecs = {
            # fmt: off
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
            # fmt: on
        }
        # pandas wants list[tuple[int, int]], 0 indexed, half open intervals.
        stations_cspecs = [(a - 1, b) for a, b in stations_colspecs.values()]

        na_values = {
            "ELEVATION": ["-999.9"],
            "COMPONENT_1": ["------"],
            "COMPONENT_2": ["------"],
            "COMPONENT_3": ["------"],
        }

        metadata = pd.read_fwf(
            self.rawdata_paths["ushcn-stations.txt"],
            colspecs=stations_cspecs,
            dtype=self.rawdata_schemas["metadata"],
            names=stations_colspecs,
            na_values=na_values,
            dtype_backend="pyarrow",
        ).set_index("COOP_ID")

        self.LOGGER.info("Removing outliers from metadata.")
        metadata = remove_outliers(metadata, self.metadata_description)

        self.LOGGER.info("Dropping completely missing rows.")
        metadata = metadata.dropna(how="all", axis="index")

        return metadata

    def _clean_timeseries(self) -> DataFrame:
        self.LOGGER.info("Creating simplified timeseries table.")
        table = self.tables["raw_timeseries"]

        self.LOGGER.info("dropping all data with raised quality flags.")
        table = table.loc[table["QFLAG"].isna()]

        self.LOGGER.info("Performing pivot operation.")
        ts = table.pivot(columns="ELEMENT", values="VALUE")
        ts.columns = ts.columns.astype("string[pyarrow]")

        self.LOGGER.info("Removing outliers from timeseries.")
        ts = remove_outliers(ts, self.timeseries_description)

        self.LOGGER.info("Dropping completely missing rows.")
        ts = ts.dropna(how="all", axis="index")

        return ts

    def _clean_raw_timeseries(self) -> DataFrame:
        # FIXME: https://github.com/pola-rs/polars/issues/8312
        # FIXME: https://github.com/apache/arrow/issues/33404
        warnings.warn(
            "This can take a while to run. Consider using the Modin backend."
            " Refactor if read_fwf becomes available in polars or pyarrow.",
            stacklevel=2,
        )

        import pyarrow.csv as csv

        # table = pl.read_csv(self.rawdata_paths["us.txt.gz"], separator="\n")
        table = csv.read_csv(
            self.rawdata_paths["us.txt.gz"],
            # compression="gzip",
        )
        print(table.schema)

        # column: (start, stop)
        colspecs: dict[str | tuple[str, int], tuple[int, int]] = {
            "COOP_ID": (1, 6),
            "YEAR": (7, 10),
            "MONTH": (11, 12),
            "ELEMENT": (13, 16),
        }

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
        ELEMENTS_DTYPE = pd.CategoricalDtype(("PRCP", "SNOW", "SNWD", "TMAX", "TMIN"))
        VALUES_DTYPE = "int16[pyarrow]"

        updated_dtypes = {
            # fmt: off
            "COOP_ID" : "int32[pyarrow]",
            "YEAR"    : "int16[pyarrow]",
            "MONTH"   : "int8[pyarrow]",
            "ELEMENT" : ELEMENTS_DTYPE,
            "VALUE"   : VALUES_DTYPE,
            "MFLAG"   : MFLAGS_DTYPE,
            "QFLAG"   : QFLAGS_DTYPE,
            "SFLAG"   : SFLAGS_DTYPE,
            # fmt: on
        }

        # dtypes but with the same index as colspec.
        base_dtypes = self.rawdata_schemas["timeseries"]
        column_dtypes = {
            key: base_dtypes[key[0]] if isinstance(key, tuple) else base_dtypes[key]
            for key in colspecs
        }

        # per column values to be interpreted as nan
        na_values = {("VALUE", k): ["-9999"] for k in range(1, 32)}

        self.LOGGER.info("Loading main file...")
        ds = pd.read_fwf(
            self.rawdata_paths["us.txt.gz"],
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
            DataFrame(data_cols, columns=["VAR", "DAY"]).astype({
                "VAR": "string[pyarrow]", "DAY": "int8[pyarrow]"
            })
        )

        self.LOGGER.info("Stacking on FLAGS and VALUES columns...")
        # stack on day, this will collapse (VALUE1, ..., VALUE31) into a single VALUE column.
        data = (
            data.stack(level="DAY", dropna=False)
            .reset_index(level="DAY")
            .astype({  # correct dtypes after stacking operation
                "DAY": "int8[pyarrow]",
                "VALUE": VALUES_DTYPE,
                "MFLAG": MFLAGS_DTYPE,
                "QFLAG": QFLAGS_DTYPE,
                "SFLAG": SFLAGS_DTYPE,
            })
        )

        self.LOGGER.info("Merging on ID columns...")
        data = data.join(index, how="inner").astype(updated_dtypes)

        self.LOGGER.info("Creating time index...")
        date_cols = ["YEAR", "MONTH", "DAY"]
        data = (
            data.assign(DATE=pd.to_datetime(data[date_cols], errors="coerce"))
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
