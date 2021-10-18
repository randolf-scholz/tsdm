r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "USHCN",
    "USHCN_SmallChunkedSporadic",
]

import gzip
import logging
import os
import sys
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Literal, Union

import pandas
from pandas import DataFrame

from tsdm.datasets.dataset import BaseDataset

LOGGER = logging.getLogger(__name__)  # noqa

try:
    os.environ["MODIN_ENGINE"] = "ray"
    from modin import pandas as pd
except ImportError as e:
    LOGGER.warning("Modin not found, falling back to pandas! %s", e)
    pd = pandas
else:
    import ray


class USHCN_SmallChunkedSporadic(BaseDataset):
    r"""Preprocessed subset of the USHCN climate dataset used by De Brouwer et. al.

    References
    ----------
    - | `GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series
        <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>`_
      | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves
      | `Advances in Neural Information Processing Systems 2019
        <https://proceedings.neurips.cc/paper/2019>`_
    """

    url = (
        "https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/"
        + "master/gru_ode_bayes/datasets/Climate/small_chunked_sporadic.csv"
    )

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def dataset_file(cls) -> Path:
        r"""Location where dataset is stored."""
        return cls.dataset_path.joinpath("SmallChunkedSporadic.feather")  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def rawdata_file(cls) -> Path:
        r"""Location where raw dataset is stored."""
        return cls.rawdata_path.joinpath("small_chunked_sporadic.csv")  # type: ignore[attr-defined]

    @classmethod
    def clean(cls):
        r"""Clean an already downloaded raw dataset and stores it in hdf5 format."""
        LOGGER.info("Finished extracting dataset '%s'", cls.__name__)
        dtypes = {
            "ID": pandas.UInt16Dtype(),
            "Time": pandas.Float32Dtype(),
            "Value_0": pandas.Float32Dtype(),
            "Value_1": pandas.Float32Dtype(),
            "Value_2": pandas.Float32Dtype(),
            "Value_3": pandas.Float32Dtype(),
            "Value_4": pandas.Float32Dtype(),
            "Mask_0": pandas.BooleanDtype(),
            "Mask_1": pandas.BooleanDtype(),
            "Mask_2": pandas.BooleanDtype(),
            "Mask_3": pandas.BooleanDtype(),
            "Mask_4": pandas.BooleanDtype(),
        }
        df = pandas.read_csv(cls.rawdata_file, dtype=dtypes)
        df.to_feather(cls.dataset_file)
        LOGGER.info("Finished cleaning dataset '%s'", cls.__name__)

    @classmethod
    def load(cls) -> DataFrame:
        r"""Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        df = pandas.read_feather(cls.dataset_file)
        df = DataFrame(df)
        return df


class USHCN(BaseDataset):
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
    """  # pylint: disable=line-too-long # noqa

    url = "https://cdiac.ess-dive.lbl.gov/ftp/ushcn_daily/"
    info_url = "https://cdiac.ess-dive.lbl.gov/epubs/ndp/ushcn/daily_doc.html"
    KEY = Literal["us_daily", "states", "stations"]
    r"""The names of the DataFrames associated with this dataset."""

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def dataset_file(cls) -> Path:
        r"""Location where dataset is stored."""
        return cls.dataset_path.joinpath("us_daily.feather")  # type: ignore[attr-defined]

    @classmethod
    def load(cls, key: KEY = "us_daily") -> DataFrame:
        r"""Load the dataset from disk."""
        path = cls.dataset_path.joinpath(f"{key}.feather")  # type: ignore[attr-defined]
        if not path.exists():
            cls.clean(key)
        return pandas.read_feather(path)

    @classmethod
    def clean(cls, key: KEY = "us_daily"):
        r"""Create the DataFrames.

        Parameters
        ----------
        key: Literal["us_daily", "states", "stations"], default="us_daily"
        """
        {
            "us_daily": cls._clean_us_daily,
            "stations": cls._clean_stations,
            "states": cls._clean_states,
        }[key]()
        LOGGER.info("Finished cleaning dataset '%s'", cls.__name__)

    @classmethod  # type: ignore
    @property
    @cache
    def states(cls) -> DataFrame:
        r"""Table containing the states data."""
        return cls.load(key="states")

    @classmethod  # type: ignore
    @property
    @cache
    def stations(cls) -> DataFrame:
        r"""Table containing the states data."""
        return cls.load(key="stations")

    @classmethod
    def _clean_states(cls):
        state_dtypes = {
            "ID": pandas.CategoricalDtype(ordered=True),
            "Abbr.": pandas.CategoricalDtype(ordered=True),
            "State": pandas.StringDtype(),
        }
        states = pandas.read_csv(StringIO(STATE_CODES), sep="\t", dtype=state_dtypes)
        states.to_feather(cls.dataset_path.joinpath("states.feather"))
        LOGGER.info("%s: Finished cleaning 'states' DataFrame", cls.__name__)

    @classmethod
    def _clean_stations(cls):
        stations_file = cls.rawdata_path.joinpath("ushcn-stations.txt")
        if not stations_file.exists():
            cls.download()

        stations_colspecs = {
            "COOP_ID": (1, 6),
            "LATITUDE": (8, 15),
            "LONGITUDE": (17, 25),
            "ELEVATION": (27, 32),
            "STATE": (34, 35),
            "NAME": (37, 66),
            "COMPONENT_1": (68, 73),
            "COMPONENT_2": (75, 80),
            "COMPONENT_3": (82, 87),
            "UTC_OFFSET": (89, 90),
        }

        # pandas wants list[tuple[int, int]], 0 indexed, half open intervals.
        stations_cspecs = [(a - 1, b) for a, b in stations_colspecs.values()]

        stations_dtypes = {
            "COOP_ID": pandas.CategoricalDtype(ordered=True),
            "LATITUDE": pandas.Float32Dtype(),
            "LONGITUDE": pandas.Float32Dtype(),
            "ELEVATION": pandas.Float32Dtype(),
            "STATE": cls.states.ID.dtype,
            "NAME": pandas.StringDtype(),
            "COMPONENT_1": pandas.CategoricalDtype(ordered=True),
            "COMPONENT_2": pandas.CategoricalDtype(ordered=True),
            "COMPONENT_3": pandas.CategoricalDtype(ordered=True),
            "UTC_OFFSET": "timedelta64[h]",
        }

        stations_na_values = {
            "ELEVATION": -999.9,
            "COMPONENT_1": "------",
            "COMPONENT_2": "------",
            "COMPONENT_3": "------",
        }

        stations = pandas.read_fwf(
            stations_file,
            colspecs=stations_cspecs,
            dtype=stations_dtypes,
            names=stations_colspecs,
            na_value=stations_na_values,
        )
        stations = DataFrame(stations)  # in case TextFileReader is returned

        # Get union of all ids
        COOP_IDS = pandas.CategoricalDtype(stations.COOP_ID, ordered=True)
        stations = stations.astype(
            {
                "COOP_ID": COOP_IDS,
                "COMPONENT_1": COOP_IDS,
                "COMPONENT_2": COOP_IDS,
                "COMPONENT_3": COOP_IDS,
            }
        )
        path = cls.dataset_path.joinpath("stations.feather")
        stations.to_feather(path)
        LOGGER.info("%s: Finished cleaning 'stations' DataFrame", cls.__name__)

    @classmethod
    def _clean_us_daily(cls):
        us_daily_file_compressed = cls.rawdata_path.joinpath("us.txt.gz")
        us_daily_file = cls.rawdata_path.joinpath("us.txt")
        if not us_daily_file_compressed.exists():
            cls.download()

        if {"modin", "ray"} <= sys.modules.keys():
            LOGGER.info("Starting ray cluster.")
            ray.init(
                num_cpus=min(1, (os.cpu_count() or 0) - 2), ignore_reinit_error=True
            )

        # column: (start, stop)
        colspecs: dict[Union[str, tuple[str, int]], tuple[int, int]] = {
            "COOP_ID": (1, 6),
            "YEAR": (7, 10),
            "MONTH": (11, 12),
            "ELEMENT": (13, 16),
        }

        for k, i in enumerate(range(17, 258, 8)):
            colspecs |= {
                ("VALUE", k + 1): (i, i + 4),
                ("MFLAG", k + 1): (i + 5, i + 5),
                ("QFLAG", k + 1): (i + 6, i + 6),
                ("SFLAG", k + 1): (i + 7, i + 7),
            }

        MFLAGS = pandas.CategoricalDtype(list("BDHKLOPTW"))
        QFLAGS = pandas.CategoricalDtype(list("DGIKLMNORSTWXZ"))
        SFLAGS = pandas.CategoricalDtype(list("067ABFGHKMNRSTUWXZ"))
        ELEMENTS = pandas.CategoricalDtype(("PRCP", "SNOW", "SNWD", "TMAX", "TMIN"))

        dtypes = {
            "COOP_ID": cls.stations.COOP_ID.dtype,
            "YEAR": pandas.UInt16Dtype(),
            "MONTH": pandas.UInt8Dtype(),
            "ELEMENT": ELEMENTS,
            "VALUE": pandas.Int16Dtype(),
            "MFLAG": MFLAGS,
            "QFLAG": QFLAGS,
            "SFLAG": SFLAGS,
        }

        # dtypes but with same keys as colspec.
        dtype = {
            key: (dtypes[key[0]] if isinstance(key, tuple) else dtypes[key])
            for key in colspecs
        }

        # pandas wants list[tuple[int, int]], 0 indexed, half open intervals.
        cspec = [(a - 1, b) for a, b in colspecs.values()]

        # per column values to be interpreted as nan
        na_values = {("VALUE", k): -9999 for k in range(1, 32)}

        # TODO: rewrite once modin.pandas.read_fwf works with gzip
        with gzip.open(us_daily_file_compressed, "rb") as compressed_file:
            with open(us_daily_file, "w", encoding="utf8") as file:
                file.write(compressed_file.read().decode("utf-8"))
        LOGGER.info("%s finished decompressing main file", cls.__name__)

        ds = pd.read_fwf(
            us_daily_file,
            colspecs=cspec,
            names=colspecs,
            na_values=na_values,
            dtype=dtype,
        )
        ds = pd.DataFrame(ds)  # In case TextFileReader was returned.
        LOGGER.info("%s finished loading main file.", cls.__name__)

        # convert data part (VALUES, SFLAGS, MFLAGS, QFLAGS) to stand-alone dataframe
        id_cols = ["COOP_ID", "YEAR", "MONTH", "ELEMENT"]
        data_cols = [col for col in ds.columns if col not in id_cols]
        # Turn tuple[VALUE/FLAG, DAY] indices to multi-index:
        columns = pd.MultiIndex.from_tuples(ds[data_cols], names=["VAR", "DAY"])
        data = pd.DataFrame(ds[data_cols])
        data.columns = columns
        # TODO: use pd.DataFrame(ds[data_cols], columns=columns) once it works correctly in modin.

        # stack on day, this will collapse (VALUE1, ..., VALUE31) into a single VALUE column.
        data = data.stack(level="DAY", dropna=True).reset_index(level="DAY")
        LOGGER.info("%s finished dataframe stacking.", cls.__name__)

        # correct dtypes after stacking operation
        _dtypes = {k: v for k, v in dtypes.items() if k in data.columns} | {
            "DAY": pandas.UInt8Dtype()
        }
        data = data.astype(_dtypes)

        # recombine data columns with original data
        data = ds[id_cols].join(data, how="inner")
        LOGGER.info("%s finished dataframe merging.", cls.__name__)

        # fix column order
        data = data[
            [
                "COOP_ID",
                "YEAR",
                "MONTH",
                "DAY",
                "ELEMENT",
                "MFLAG",
                "QFLAG",
                "SFLAG",
                "VALUE",
            ]
        ]

        # optional: sorting
        data = data.sort_values(by=["YEAR", "MONTH", "DAY", "COOP_ID", "ELEMENT"])
        LOGGER.info("%s finished sorting.", cls.__name__)

        # drop old index which may contain duplicates
        data = data.reset_index(drop=True)

        data.to_feather(cls.dataset_file)
        LOGGER.info("%s: Finished cleaning 'us_daily' DataFrame", cls.__name__)


STATE_CODES = r"""
ID	Abbr.	State
01	AL	Alabama
02	AZ	Arizona
03	AR	Arkansas
04	CA	California
05	CO	Colorado
06	CT	Connecticut
07	DE	Delaware
08	FL	Florida
09	GA	Georgia
10	ID	Idaho
11	IL	Idaho
12	IN	Indiana
13	IA	Iowa
14	KS	Kansas
15	KY	Kentucky
16	LA	Louisiana
17	ME	Maine
18	MD	Maryland
19	MA	Massachusetts
20	MI	Michigan
21	MN	Minnesota
22	MS	Mississippi
23	MO	Missouri
24	MT	Montana
25	NE	Nebraska
26	NV	Nevada
27	NH	NewHampshire
28	NJ	NewJersey
29	NM	NewMexico
30	NY	NewYork
31	NC	NorthCarolina
32	ND	NorthDakota
33	OH	Ohio
34	OK	Oklahoma
35	OR	Oregon
36	PA	Pennsylvania
37	RI	RhodeIsland
38	SC	SouthCarolina
39	SD	SouthDakota
40	TN	Tennessee
41	TX	Texas
42	UT	Utah
43	VT	Vermont
44	VA	Virginia
45	WA	Washington
46	WV	WestVirginia
47	WI	Wisconsin
48	WY	Wyoming
"""
