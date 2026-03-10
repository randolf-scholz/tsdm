r"""Module defining various scalar and array types for different backends."""
# ruff: noqa: A003

import datetime as dt
from abc import ABC, abstractmethod
from typing import Literal as L

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch as pt

from tsdm.utils import timedelta as make_pd_timedelta, timestamp as make_pd_timestamp

type Array1D[_SCT: np.generic] = np.ndarray[tuple[int], np.dtype[_SCT]]
type Array2D[_SCT: np.generic] = np.ndarray[tuple[int, int], np.dtype[_SCT]]
type ArrayND[_SCT: np.generic] = np.ndarray[tuple[int, ...], np.dtype[_SCT]]


class types0d:
    class py:
        # fmt: off
        type none      = None
        type true      = L[True]
        type false     = L[False]
        type one       = L[1]
        type zero      = L[0]
        type bool      = bool
        type int       = int
        type float     = float
        type complex   = complex
        type bytes     = bytes
        type string    = str
        type tuple     = tuple
        type date      = dt.date
        type datetime  = dt.datetime
        type timedelta = dt.timedelta
        # fmt: on

    class np:
        # fmt: off
        type bool      = np.bool_
        type int       = np.int64
        type float     = np.float64
        type complex   = np.complex128
        type date      = np.datetime64
        type datetime  = np.datetime64
        type timedelta = np.timedelta64
        # fmt: on

    class pd:
        # fmt: off
        type bool      = bool
        type int       = int
        type float     = float
        type complex   = complex
        type string    = str
        type date      = pd.Timestamp
        type datetime  = pd.Timestamp
        type timedelta = pd.Timedelta
        # fmt: on

    class pl:
        # fmt: off
        type bool      = bool
        type int       = int
        type float     = float
        type string    = str
        type date      = dt.date
        type datetime  = dt.datetime
        type timedelta = dt.timedelta
        # fmt: on

    class pt:
        # fmt: off
        type bool      = pt.Tensor
        type int       = pt.Tensor
        type float     = pt.Tensor
        type complex   = pt.Tensor
        # fmt: on

    class pa:
        # fmt: off
        type bool      = pa.bool_
        type int       = pa.int64
        type float     = pa.float64
        type string    = pa.string
        type date      = pa.date32
        type datetime  = pa.timestamp
        type timedelta = pa.duration
        # fmt: on


class types1d:
    class py:
        # fmt: off
        type none      = list[types0d.py.none]
        type true      = list[types0d.py.true]
        type false     = list[types0d.py.false]
        type one       = list[types0d.py.one]
        type zero      = list[types0d.py.zero]
        type bool      = list[types0d.py.bool]
        type int       = list[types0d.py.int]
        type float     = list[types0d.py.float]
        type complex   = list[types0d.py.complex]
        type bytes     = list[types0d.py.bytes]
        type string    = list[types0d.py.string]
        type tuple     = list[types0d.py.tuple]
        type date      = list[types0d.py.date]
        type datetime  = list[types0d.py.datetime]
        type timedelta = list[types0d.py.timedelta]
        # fmt: on

    class np:
        # fmt: off
        type bool      = Array1D[types0d.np.bool     ]
        type int       = Array1D[types0d.np.int      ]
        type float     = Array1D[types0d.np.float    ]
        type complex   = Array1D[types0d.np.complex  ]
        type date      = Array1D[types0d.np.date     ]
        type datetime  = Array1D[types0d.np.datetime ]
        type timedelta = Array1D[types0d.np.timedelta]
        # fmt: on

    class pd:
        # fmt: off
        type bool      = pd.Series
        type int       = pd.Series
        type float     = pd.Series
        type complex   = pd.Series
        type string    = pd.Series
        type date      = pd.Series
        type datetime  = pd.Series
        type timedelta = pd.Series
        # fmt: on

    class pl:
        # fmt: off
        type bool      = pl.Series
        type int       = pl.Series
        type float     = pl.Series
        type string    = pl.Series
        type date      = pl.Series
        type datetime  = pl.Series
        type timedelta = pl.Series
        # fmt: on

    class pt:
        # fmt: off
        type bool    = pt.Tensor
        type int     = pt.Tensor
        type float   = pt.Tensor
        type complex = pt.Tensor
        # fmt: on


class types2d:
    class py:
        # fmt: off
        type none      = list[list[types0d.py.none]]
        type true      = list[list[types0d.py.true]]
        type false     = list[list[types0d.py.false]]
        type one       = list[list[types0d.py.one]]
        type zero      = list[list[types0d.py.zero]]
        type bool      = list[list[types0d.py.bool]]
        type int       = list[list[types0d.py.int]]
        type float     = list[list[types0d.py.float]]
        type complex   = list[list[types0d.py.complex]]
        type bytes     = list[list[types0d.py.bytes]]
        type string    = list[list[types0d.py.string]]
        type tuple     = list[list[types0d.py.tuple]]
        type date      = list[list[types0d.py.date]]
        type datetime  = list[list[types0d.py.datetime]]
        type timedelta = list[list[types0d.py.timedelta]]
        # fmt: on

    class np:
        # fmt: off
        type bool      = Array2D[types0d.np.bool]
        type int       = Array2D[types0d.np.int]
        type float     = Array2D[types0d.np.float]
        type complex   = Array2D[types0d.np.complex]
        type date      = Array2D[types0d.np.date]
        type datetime  = Array2D[types0d.np.datetime]
        type timedelta = Array2D[types0d.np.timedelta]
        # fmt: on

    class pd:
        # fmt: off
        type bool      = pd.DataFrame
        type int       = pd.DataFrame
        type float     = pd.DataFrame
        type complex   = pd.DataFrame
        type date      = pd.DataFrame
        type datetime  = pd.DataFrame
        type timedelta = pd.DataFrame
        # fmt: on

    class pl:
        # fmt: off
        type bool      = pl.DataFrame
        type int       = pl.DataFrame
        type float     = pl.DataFrame
        type string    = pl.DataFrame
        type date      = pl.DataFrame
        type datetime  = pl.DataFrame
        type timedelta = pl.DataFrame
        # fmt: on

    class pt:
        # fmt: off
        type bool    = pt.Tensor
        type int     = pt.Tensor
        type float   = pt.Tensor
        type complex = pt.Tensor
        # fmt: on


class typesNd:
    class py:
        # fmt: off
        type none      = list[list[list[types0d.py.none]]]
        type bool      = list[list[list[types0d.py.bool]]]
        type int       = list[list[list[types0d.py.int]]]
        type float     = list[list[list[types0d.py.float]]]
        type complex   = list[list[list[types0d.py.complex]]]
        type string    = list[list[list[types0d.py.string]]]
        type bytes     = list[list[list[types0d.py.bytes]]]
        type date      = list[list[list[types0d.py.date]]]
        type datetime  = list[list[list[types0d.py.datetime]]]
        type timedelta = list[list[list[types0d.py.timedelta]]]
        # fmt: on

    class np:
        # fmt: off
        type bool      = ArrayND[types0d.np.bool     ]
        type int       = ArrayND[types0d.np.int      ]
        type float     = ArrayND[types0d.np.float    ]
        type complex   = ArrayND[types0d.np.complex  ]
        type date      = ArrayND[types0d.np.date     ]
        type datetime  = ArrayND[types0d.np.datetime ]
        type timedelta = ArrayND[types0d.np.timedelta]
        # fmt: on

    class pt:
        # fmt: off
        type bool    = pt.Tensor
        type int     = pt.Tensor
        type float   = pt.Tensor
        type complex = pt.Tensor
        # fmt: on


# fmt: off
# 0d date (scalars)
NONE      : types0d.py.none      = None
TRUE      : types0d.py.true      = True
FALSE     : types0d.py.false     = False
ONE       : types0d.py.one       = 1
ZERO      : types0d.py.zero      = 0
BOOL      : types0d.py.bool      = bool(1)
INT       : types0d.py.int       = int(2)
FLOAT     : types0d.py.float     = float(1.23)
COMPLEX   : types0d.py.complex   = complex(1.2 + 2.3j)
BYTES     : types0d.py.bytes     = bytes(b"test")
STRING    : types0d.py.string    = str("test")
TUPLE     : types0d.py.tuple     = tuple([1, 2, 3])  # noqa: C409
DATE      : types0d.py.date      = dt.date(2021, 1, 2)
DATETIME  : types0d.py.datetime  = dt.datetime(2021, 1, 2, 3, 4, 5)
TIMEDELTA : types0d.py.timedelta = dt.timedelta(days=1, hours=2, seconds=4)
# 1d data
NONE_1D      : types1d.py.none      = [None, None, None, None]
TRUE_1D      : types1d.py.true      = [True, True, True, True]
FALSE_1D     : types1d.py.false     = [False, False, False, False]
BOOL_1D      : types1d.py.bool      = [False, False, True, False]
INT_1D       : types1d.py.int       = [1, 2, 3, 4]
FLOAT_1D     : types1d.py.float     = [1.1, 2.2, 3.3, 4.4]
COMPLEX_1D   : types1d.py.complex   = [1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j, 7.0 + 8.0j]
STRING_1D    : types1d.py.string    = ["a", "b", "c", "d"]
BYTES_1D     : types1d.py.bytes     = [b"a", b"b", b"c", b"d"]
DATE_1D      : types1d.py.date      = [dt.date(2021, 1, 1), dt.date(2021, 1, 2), dt.date(2021, 1, 3), dt.date(2021, 1, 4)]  # noqa: E501
DATETIME_1D  : types1d.py.datetime  = [dt.datetime(2021, 1, 1, 0, 0, 0), dt.datetime(2021, 1, 2, 12, 0, 0), dt.datetime(2021, 1, 3, 18, 30, 0), dt.datetime(2021, 1, 4, 23, 59, 59)]  # noqa: E501
TIMEDELTA_1D : types1d.py.timedelta = [dt.timedelta(days=1), dt.timedelta(days=2, hours=3), dt.timedelta(hours=5, minutes=30), dt.timedelta(weeks=1)]  # noqa: E501
# 2d data
NONE_2D      : types2d.py.none      = [NONE_1D     , NONE_1D     , NONE_1D     , NONE_1D     ]
BOOL_2D      : types2d.py.bool      = [BOOL_1D     , BOOL_1D     , BOOL_1D     , BOOL_1D     ]
INT_2D       : types2d.py.int       = [INT_1D      , INT_1D      , INT_1D      , INT_1D      ]
FLOAT_2D     : types2d.py.float     = [FLOAT_1D    , FLOAT_1D    , FLOAT_1D    , FLOAT_1D    ]
COMPLEX_2D   : types2d.py.complex   = [COMPLEX_1D  , COMPLEX_1D  , COMPLEX_1D  , COMPLEX_1D  ]
STRING_2D    : types2d.py.string    = [STRING_1D   , STRING_1D   , STRING_1D   , STRING_1D   ]
BYTES_2D     : types2d.py.bytes     = [BYTES_1D    , BYTES_1D    , BYTES_1D    , BYTES_1D    ]
DATE_2D      : types2d.py.date      = [DATE_1D     , DATE_1D     , DATE_1D     , DATE_1D     ]
DATETIME_2D  : types2d.py.datetime  = [DATETIME_1D , DATETIME_1D , DATETIME_1D , DATETIME_1D ]
TIMEDELTA_2D : types2d.py.timedelta = [TIMEDELTA_1D, TIMEDELTA_1D, TIMEDELTA_1D, TIMEDELTA_1D]
# Nd data
ONE_ND       : typesNd.py.none      = [NONE_2D     , NONE_2D     ]
BOOL_ND      : typesNd.py.bool      = [BOOL_2D     , BOOL_2D     ]
INT_ND       : typesNd.py.int       = [INT_2D      , INT_2D      ]
FLOAT_ND     : typesNd.py.float     = [FLOAT_2D    , FLOAT_2D    ]
COMPLEX_ND   : typesNd.py.complex   = [COMPLEX_2D  , COMPLEX_2D  ]
STRING_ND    : typesNd.py.string    = [STRING_2D   , STRING_2D   ]
BYTES_ND     : typesNd.py.bytes     = [BYTES_2D    , BYTES_2D    ]
DATE_ND      : typesNd.py.date      = [DATE_2D     , DATE_2D     ]
DATETIME_ND  : typesNd.py.datetime  = [DATETIME_2D , DATETIME_2D ]
TIMEDELTA_ND : typesNd.py.timedelta = [TIMEDELTA_2D, TIMEDELTA_2D]
# tabular data
DICT_INT = {
    "a": [1, 2, 3, 4],
    "b": [5, 6, 7, 8],
    "c": [9, 0, 1, 2],
}
DICT_FLOAT = {
    "x": [1.1, 2.2, 3.3, 4.4],
    "y": [5.5, 6.6, 7.7, 8.8],
    "z": [9.9, 0.0, 1.1, 2.2],
}
DICT_MIXED = {
    "label": ["a", "b", "c", "d"],
    "x": [111, 222, 333, 444],
    "y": [5.5, 6.6, 7.7, 8.8],
}
# fmt: on


class DTYPES:
    class NP:
        # fmt: off
        BOOL     : np.dtype[types0d.np.bool]      = np.dtype(np.bool_)
        INT      : np.dtype[types0d.np.int]       = np.dtype(np.int64)
        FLOAT    : np.dtype[types0d.np.float]     = np.dtype(np.float64)
        COMPLEX  : np.dtype[types0d.np.complex]   = np.dtype(np.complex128)
        DATE     : np.dtype[types0d.np.date]      = np.dtype("datetime64[D]")
        DATETIME : np.dtype[types0d.np.datetime]  = np.dtype("datetime64[ms]")
        TIMEDELTA: np.dtype[types0d.np.timedelta] = np.dtype("timedelta64[ms]")
        # fmt: on

    class PD_NP:
        # fmt: off
        BOOL      : np.dtype[types0d.np.bool]      = np.dtype(np.bool_)
        INT       : np.dtype[types0d.np.int]       = np.dtype(np.int64)
        FLOAT     : np.dtype[types0d.np.float]     = np.dtype(np.float64)
        COMPLEX   : np.dtype[types0d.np.complex]   = np.dtype(np.complex128)
        STRING    : pd.StringDtype                 = pd.StringDtype()
        DATE      : np.dtype[types0d.np.date]      = np.dtype("datetime64[D]")
        DATETIME  : np.dtype[types0d.np.datetime]  = np.dtype("datetime64[ms]")
        TIMEDELTA : np.dtype[types0d.np.timedelta] = np.dtype("timedelta64[ms]")
        # fmt: on

    class PD_PA:
        # fmt: off
        BOOL     : pd.ArrowDtype = pd.ArrowDtype(pa.bool_())
        INT      : pd.ArrowDtype = pd.ArrowDtype(pa.int64())
        FLOAT    : pd.ArrowDtype = pd.ArrowDtype(pa.float64())
        STRING   : pd.ArrowDtype = pd.ArrowDtype(pa.string())
        DATE     : pd.ArrowDtype = pd.ArrowDtype(pa.date32())
        DATETIME : pd.ArrowDtype = pd.ArrowDtype(pa.timestamp("ms"))
        TIMEDELTA: pd.ArrowDtype = pd.ArrowDtype(pa.duration("ms"))
        # fmt: on

    class PL:
        # fmt: off
        BOOL      = pl.Boolean
        INT       = pl.Int64
        FLOAT     = pl.Float64
        STRING    = pl.String
        DATE      = pl.Date
        DATETIME  = pl.Datetime
        TIMEDELTA = pl.Duration
        # fmt: on

    class PT:
        # fmt: off
        BOOL      = pt.bool
        INT       = pt.int64
        FLOAT     = pt.float64
        COMPLEX   = pt.complex128
        # fmt: on

    class PA:
        # fmt: off
        BOOL      = pa.bool_()
        INT       = pa.int64()
        FLOAT     = pa.float64()
        STRING    = pa.string()
        DATE      = pa.date32()
        DATETIME  = pa.timestamp("ms")
        TIMEDELTA = pa.duration("ms")
        # fmt: on


class PYTHON:
    class SCALARS:
        # fmt: off
        NONE      : types0d.py.none      = NONE
        TRUE      : types0d.py.true      = TRUE
        FALSE     : types0d.py.false     = FALSE
        ONE       : types0d.py.one       = ONE
        ZERO      : types0d.py.zero      = ZERO
        BOOL      : types0d.py.bool      = BOOL
        INT       : types0d.py.int       = INT
        FLOAT     : types0d.py.float     = FLOAT
        COMPLEX   : types0d.py.complex   = COMPLEX
        BYTES     : types0d.py.bytes     = BYTES
        STRING    : types0d.py.string    = STRING
        TUPLE     : types0d.py.tuple     = TUPLE
        DATE      : types0d.py.date      = DATE
        DATETIME  : types0d.py.datetime  = DATETIME
        TIMEDELTA : types0d.py.timedelta = TIMEDELTA
        # fmt: on

    class SERIES:
        # fmt: off
        NONE      = NONE_1D
        BOOL      = BOOL_1D
        INT       = INT_1D
        FLOAT     = FLOAT_1D
        COMPLEX   = COMPLEX_1D
        STRING    = STRING_1D
        BYTES     = BYTES_1D
        DATE      = DATE_1D
        DATETIME  = DATETIME_1D
        TIMEDELTA = TIMEDELTA_1D
        # fmt: on

    class TABLES:
        FLOAT = DICT_FLOAT
        MIXED = DICT_MIXED


class PANDAS_PA:
    class SERIES_NOINDEX:
        # fmt: off
        BOOL      = pd.Series(BOOL_1D      , dtype=DTYPES.PD_PA.BOOL)
        INT       = pd.Series(INT_1D       , dtype=DTYPES.PD_PA.INT)
        FLOAT     = pd.Series(FLOAT_1D     , dtype=DTYPES.PD_PA.FLOAT)
        # COMPLEX   = pd.Series(COMPLEX_1D   , dtype=DTYPES.PD_PA.COMPLEX)
        STRING    = pd.Series(STRING_1D    , dtype=DTYPES.PD_PA.STRING)
        DATE      = pd.Series(DATE_1D      , dtype=DTYPES.PD_PA.DATE)
        DATETIME  = pd.Series(DATETIME_1D  , dtype=DTYPES.PD_PA.DATETIME)
        TIMEDELTA = pd.Series(TIMEDELTA_1D , dtype=DTYPES.PD_PA.TIMEDELTA)
        # fmt: on

    class SERIES_INDEX:
        # fmt: off
        BOOL      = pd.Series(BOOL_1D     , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.BOOL)
        INT       = pd.Series(INT_1D      , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.INT)
        FLOAT     = pd.Series(FLOAT_1D    , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.FLOAT)
        # COMPLEX   = pd.Series(COMPLEX_1D  , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.COMPLEX)
        STRING    = pd.Series(STRING_1D   , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.STRING)
        DATE      = pd.Series(DATE_1D     , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.DATE)
        DATETIME  = pd.Series(DATETIME_1D , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.DATETIME)
        TIMEDELTA = pd.Series(TIMEDELTA_1D, index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_PA.TIMEDELTA)
        # fmt: on

    class TABLES:
        FLOAT = pd.DataFrame(DICT_FLOAT).astype(
            {
                "x": DTYPES.PD_PA.FLOAT,
                "y": DTYPES.PD_PA.FLOAT,
                "z": DTYPES.PD_PA.FLOAT,
            }
        )
        MIXED = pd.DataFrame(DICT_MIXED).astype(
            {
                "label": DTYPES.PD_PA.STRING,
                "x": DTYPES.PD_PA.INT,
                "y": DTYPES.PD_PA.FLOAT,
            }
        )


class PANDAS:
    class SCALARS:
        # fmt: off
        DATETIME  : types0d.pd.datetime  = make_pd_timestamp(DATETIME)
        TIMEDELTA : types0d.pd.timedelta = make_pd_timedelta(TIMEDELTA)
        # fmt: on


class PANDAS_NP:
    class SERIES_NOINDEX:
        # fmt: off
        BOOL      = pd.Series(BOOL_1D      , dtype=DTYPES.PD_NP.BOOL)
        INT       = pd.Series(INT_1D       , dtype=DTYPES.PD_NP.INT)
        FLOAT     = pd.Series(FLOAT_1D     , dtype=DTYPES.PD_NP.FLOAT)
        COMPLEX   = pd.Series(COMPLEX_1D   , dtype=DTYPES.PD_NP.COMPLEX)
        STRING    = pd.Series(STRING_1D    , dtype=DTYPES.PD_NP.STRING)
        # DATE      = pd.Series(DATE_1D      , dtype=DTYPES.PD_NP.DATE)
        DATETIME  = pd.Series(DATETIME_1D  , dtype=DTYPES.PD_NP.DATETIME)
        TIMEDELTA = pd.Series(TIMEDELTA_1D , dtype=DTYPES.PD_NP.TIMEDELTA)
        # fmt: on

    class SERIES_INDEX:
        # fmt: off
        BOOL      = pd.Series(BOOL_1D     , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.BOOL)
        INT       = pd.Series(INT_1D      , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.INT)
        FLOAT     = pd.Series(FLOAT_1D    , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.FLOAT)
        COMPLEX   = pd.Series(COMPLEX_1D  , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.COMPLEX)
        STRING    = pd.Series(STRING_1D   , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.STRING)
        # DATE      = pd.Series(DATE_1D     , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.DATE)
        DATETIME  = pd.Series(DATETIME_1D , index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.DATETIME)
        TIMEDELTA = pd.Series(TIMEDELTA_1D, index=pd.Index(DATETIME_1D), dtype=DTYPES.PD_NP.TIMEDELTA)
        # fmt: on

    class TABLES:
        FLOAT = pd.DataFrame(DICT_FLOAT).astype(
            {
                "x": DTYPES.PD_NP.FLOAT,
                "y": DTYPES.PD_NP.FLOAT,
                "z": DTYPES.PD_NP.FLOAT,
            }
        )
        MIXED = pd.DataFrame(DICT_MIXED).astype(
            {
                "label": DTYPES.PD_NP.STRING,
                "x": DTYPES.PD_NP.INT,
                "y": DTYPES.PD_NP.FLOAT,
            }
        )


class NUMPY:
    class SCALARS:
        # fmt: off
        BOOL      : types0d.np.bool      = np.bool_(BOOL)
        INT       : types0d.np.int       = np.int64(INT)
        FLOAT     : types0d.np.float     = np.float64(FLOAT)
        COMPLEX   : types0d.np.complex   = np.complex128(COMPLEX)
        DATE      : types0d.np.date      = np.datetime64(DATE)
        DATETIME  : types0d.np.datetime  = np.datetime64(DATETIME, "ms")
        TIMEDELTA : types0d.np.timedelta = np.timedelta64(TIMEDELTA, "ms")
        # fmt: on

    class ARRAY_1D:
        # fmt: off
        BOOL      = np.array(BOOL_1D      , dtype=DTYPES.NP.BOOL)
        INT       = np.array(INT_1D       , dtype=DTYPES.NP.INT)
        FLOAT     = np.array(FLOAT_1D     , dtype=DTYPES.NP.FLOAT)
        COMPLEX   = np.array(COMPLEX_1D   , dtype=DTYPES.NP.COMPLEX)
        DATE      = np.array(DATE_1D      , dtype=DTYPES.NP.DATE)
        DATETIME  = np.array(DATETIME_1D  , dtype=DTYPES.NP.DATETIME)
        TIMEDELTA = np.array(TIMEDELTA_1D , dtype=DTYPES.NP.TIMEDELTA)
        # fmt: on

    class ARRAY_2D:
        # fmt: off
        BOOL      = np.array(BOOL_2D      , dtype=DTYPES.NP.BOOL)
        INT       = np.array(INT_2D       , dtype=DTYPES.NP.INT)
        FLOAT     = np.array(FLOAT_2D     , dtype=DTYPES.NP.FLOAT)
        COMPLEX   = np.array(COMPLEX_2D   , dtype=DTYPES.NP.COMPLEX)
        DATE      = np.array(DATE_2D      , dtype=DTYPES.NP.DATE)
        DATETIME  = np.array(DATETIME_2D  , dtype=DTYPES.NP.DATETIME)
        TIMEDELTA = np.array(TIMEDELTA_2D , dtype=DTYPES.NP.TIMEDELTA)
        # fmt: on

    class ARRAY_ND:
        # fmt: off
        BOOL      = np.array(BOOL_ND      , dtype=DTYPES.NP.BOOL)
        INT       = np.array(INT_ND       , dtype=DTYPES.NP.INT)
        FLOAT     = np.array(FLOAT_ND     , dtype=DTYPES.NP.FLOAT)
        COMPLEX   = np.array(COMPLEX_ND   , dtype=DTYPES.NP.COMPLEX)
        DATE      = np.array(DATE_ND      , dtype=DTYPES.NP.DATE)
        DATETIME  = np.array(DATETIME_ND  , dtype=DTYPES.NP.DATETIME)
        TIMEDELTA = np.array(TIMEDELTA_ND , dtype=DTYPES.NP.TIMEDELTA)
        # fmt: on


class TORCH:
    class SCALARS:
        # fmt: off
        BOOL      : types0d.pt.bool    = pt.tensor(BOOL, dtype=pt.bool)
        INT       : types0d.pt.int     = pt.tensor(INT, dtype=pt.int64)
        FLOAT     : types0d.pt.float   = pt.tensor(FLOAT, dtype=pt.float64)
        COMPLEX   : types0d.pt.complex = pt.tensor(COMPLEX, dtype=pt.complex128)
        # fmt: on

    class ARRAY_1D:
        # fmt: off
        BOOL    = pt.tensor(BOOL_1D   , dtype=DTYPES.PT.BOOL)
        INT     = pt.tensor(INT_1D    , dtype=DTYPES.PT.INT)
        FLOAT   = pt.tensor(FLOAT_1D  , dtype=DTYPES.PT.FLOAT)
        COMPLEX = pt.tensor(COMPLEX_1D, dtype=DTYPES.PT.COMPLEX)
        # fmt: on

    class ARRAY_2D:
        # fmt: off
        BOOL    = pt.tensor(BOOL_2D   , dtype=DTYPES.PT.BOOL)
        INT     = pt.tensor(INT_2D    , dtype=DTYPES.PT.INT)
        FLOAT   = pt.tensor(FLOAT_2D  , dtype=DTYPES.PT.FLOAT)
        COMPLEX = pt.tensor(COMPLEX_2D, dtype=DTYPES.PT.COMPLEX)
        # fmt: on

    class ARRAY_ND:
        # fmt: off
        BOOL    = pt.tensor(BOOL_ND   , dtype=DTYPES.PT.BOOL)
        INT     = pt.tensor(INT_ND    , dtype=DTYPES.PT.INT)
        FLOAT   = pt.tensor(FLOAT_ND  , dtype=DTYPES.PT.FLOAT)
        COMPLEX = pt.tensor(COMPLEX_ND, dtype=DTYPES.PT.COMPLEX)
        # fmt: on


class POLARS:
    class SERIES:
        # fmt: off
        BOOL      = pl.Series(BOOL_1D     , dtype=DTYPES.PL.BOOL)
        INT       = pl.Series(INT_1D      , dtype=DTYPES.PL.INT)
        FLOAT     = pl.Series(FLOAT_1D    , dtype=DTYPES.PL.FLOAT)
        STRING    = pl.Series(STRING_1D   , dtype=DTYPES.PL.STRING)
        DATE      = pl.Series(DATE_1D     , dtype=DTYPES.PL.DATE)
        DATETIME  = pl.Series(DATETIME_1D , dtype=DTYPES.PL.DATETIME)
        TIMEDELTA = pl.Series(TIMEDELTA_1D, dtype=DTYPES.PL.TIMEDELTA)
        # fmt: on

    class TABLES:
        FLOAT = pl.DataFrame(DICT_FLOAT)
        MIXED = pl.DataFrame(DICT_MIXED)


class PYARROW:
    class SCALARS:
        # fmt: off
        BOOL      : types0d.pa.bool      = pa.scalar(BOOL, type=DTYPES.PA.BOOL)
        INT       : types0d.pa.int       = pa.scalar(INT, type=DTYPES.PA.INT)
        FLOAT     : types0d.pa.float     = pa.scalar(FLOAT, type=DTYPES.PA.FLOAT)
        STRING    : types0d.pa.string    = pa.scalar(STRING, type=DTYPES.PA.STRING)
        DATE      : types0d.pa.date      = pa.scalar(DATE, type=DTYPES.PA.DATE)
        DATETIME  : types0d.pa.datetime  = pa.scalar(DATETIME, type=DTYPES.PA.DATETIME)
        TIMEDELTA : types0d.pa.timedelta = pa.scalar(TIMEDELTA, type=DTYPES.PA.TIMEDELTA)
        # fmt: on

    class SERIES:
        # fmt: off
        BOOL      = pa.array(BOOL_1D     , type=DTYPES.PA.BOOL)
        INT       = pa.array(INT_1D      , type=DTYPES.PA.INT)
        FLOAT     = pa.array(FLOAT_1D    , type=DTYPES.PA.FLOAT)
        STRING    = pa.array(STRING_1D   , type=DTYPES.PA.STRING)
        DATE      = pa.array(DATE_1D     , type=DTYPES.PA.DATE)
        DATETIME  = pa.array(DATETIME_1D , type=DTYPES.PA.DATETIME)
        TIMEDELTA = pa.array(TIMEDELTA_1D, type=DTYPES.PA.TIMEDELTA)
        # fmt: on

    class TABLES:
        FLOAT = pa.table(DICT_FLOAT)
        MIXED = pa.table(DICT_MIXED)


class SCALARS:
    PY = PYTHON.SCALARS
    NP = NUMPY.SCALARS
    PD = PANDAS.SCALARS
    PT = TORCH.SCALARS

    class BOOL:
        PY = PYTHON.SCALARS.BOOL
        NP = NUMPY.SCALARS.BOOL
        PT = TORCH.SCALARS.BOOL
        PA = PYARROW.SCALARS.BOOL

    class INT:
        PY = PYTHON.SCALARS.INT
        NP = NUMPY.SCALARS.INT
        PT = TORCH.SCALARS.INT
        PA = PYARROW.SCALARS.INT

    class FLOAT:
        PY = PYTHON.SCALARS.FLOAT
        NP = NUMPY.SCALARS.FLOAT
        PT = TORCH.SCALARS.FLOAT
        PA = PYARROW.SCALARS.FLOAT

    class COMPLEX:
        PY = PYTHON.SCALARS.COMPLEX
        NP = NUMPY.SCALARS.COMPLEX
        PT = TORCH.SCALARS.COMPLEX

    class DATE:
        PY = PYTHON.SCALARS.DATE
        NP = NUMPY.SCALARS.DATE
        PA = PYARROW.SCALARS.DATE

    class DATETIME:
        PY = PYTHON.SCALARS.DATETIME
        NP = NUMPY.SCALARS.DATETIME
        PD = PANDAS.SCALARS.DATETIME
        PA = PYARROW.SCALARS.DATETIME

    class TIMEDELTA:
        PY = PYTHON.SCALARS.TIMEDELTA
        NP = NUMPY.SCALARS.TIMEDELTA
        PD = PANDAS.SCALARS.TIMEDELTA
        PA = PYARROW.SCALARS.TIMEDELTA

    class STRING:
        PY = PYTHON.SCALARS.STRING
        PA = PYARROW.SCALARS.STRING


class SERIES:
    PY = PYTHON.SERIES
    NP = NUMPY.ARRAY_1D
    PD_NP = PANDAS_NP.SERIES_NOINDEX
    PD_PA = PANDAS_PA.SERIES_NOINDEX
    PL = POLARS.SERIES
    PT = TORCH.ARRAY_1D
    PA = PYARROW.SERIES

    class BOOL:
        NP = NUMPY.ARRAY_1D.BOOL
        PT = TORCH.ARRAY_1D.BOOL
        PY = PYTHON.SERIES.BOOL
        PD_NP = PANDAS_NP.SERIES_NOINDEX.BOOL
        PD_PA = PANDAS_PA.SERIES_NOINDEX.BOOL
        PL = POLARS.SERIES.BOOL
        PA = PYARROW.SERIES.BOOL

    class INT:
        NP = NUMPY.ARRAY_1D.INT
        PT = TORCH.ARRAY_1D.INT
        PY = PYTHON.SERIES.INT
        PD_NP = PANDAS_NP.SERIES_NOINDEX.INT
        PD_PA = PANDAS_PA.SERIES_NOINDEX.INT
        PL = POLARS.SERIES.INT
        PA = PYARROW.SERIES.INT

    class FLOAT:
        NP = NUMPY.ARRAY_1D.FLOAT
        PT = TORCH.ARRAY_1D.FLOAT
        PY = PYTHON.SERIES.FLOAT
        PD_NP = PANDAS_NP.SERIES_NOINDEX.FLOAT
        PD_PA = PANDAS_PA.SERIES_NOINDEX.FLOAT
        PL = POLARS.SERIES.FLOAT
        PA = PYARROW.SERIES.FLOAT

    class COMPLEX:
        NP = NUMPY.ARRAY_1D.COMPLEX
        PT = TORCH.ARRAY_1D.COMPLEX
        PY = PYTHON.SERIES.COMPLEX
        PD_NP = PANDAS_NP.SERIES_NOINDEX.COMPLEX

    class DATE:
        NP = NUMPY.ARRAY_1D.DATE
        PY = PYTHON.SERIES.DATE
        # PD_NP = PANDAS_NP.SERIES_NOINDEX.DATE
        PD_PA = PANDAS_PA.SERIES_NOINDEX.DATE
        PL = POLARS.SERIES.DATE
        PA = PYARROW.SERIES.DATE

    class DATETIME:
        NP = NUMPY.ARRAY_1D.DATETIME
        PY = PYTHON.SERIES.DATETIME
        PD_NP = PANDAS_NP.SERIES_NOINDEX.DATETIME
        PD_PA = PANDAS_PA.SERIES_NOINDEX.DATETIME
        PL = POLARS.SERIES.DATETIME
        PA = PYARROW.SERIES.DATETIME

    class TIMEDELTA:
        NP = NUMPY.ARRAY_1D.TIMEDELTA
        PY = PYTHON.SERIES.TIMEDELTA
        PD_NP = PANDAS_NP.SERIES_NOINDEX.TIMEDELTA
        PD_PA = PANDAS_PA.SERIES_NOINDEX.TIMEDELTA
        PL = POLARS.SERIES.TIMEDELTA
        PA = PYARROW.SERIES.TIMEDELTA


ARRAYS1D = SERIES  # alias


class ARRAYS2D:
    NP = NUMPY.ARRAY_2D
    PT = TORCH.ARRAY_2D

    class BOOL:
        NP = NUMPY.ARRAY_2D.BOOL
        PT = TORCH.ARRAY_2D.BOOL

    class INT:
        NP = NUMPY.ARRAY_2D.INT
        PT = TORCH.ARRAY_2D.INT

    class FLOAT:
        NP = NUMPY.ARRAY_2D.FLOAT
        PT = TORCH.ARRAY_2D.FLOAT

    class COMPLEX:
        NP = NUMPY.ARRAY_2D.COMPLEX
        PT = TORCH.ARRAY_2D.COMPLEX


class TABLES:
    PY = PYTHON.TABLES
    PA = PYARROW.TABLES
    PL = POLARS.TABLES
    PD_NP = PANDAS_NP.TABLES
    PD_PA = PANDAS_PA.TABLES

    class FLOAT:
        PA = PYARROW.TABLES.FLOAT
        PL = POLARS.TABLES.FLOAT
        PD_NP = PANDAS_NP.TABLES.FLOAT
        PD_PA = PANDAS_PA.TABLES.FLOAT

    class MIXED:
        PA = PYARROW.TABLES.MIXED
        PL = POLARS.TABLES.MIXED
        PD_NP = PANDAS_NP.TABLES.MIXED
        PD_PA = PANDAS_PA.TABLES.MIXED


class TestAssignableSuite(ABC):
    @abstractmethod
    def test_no_generic(self): ...
    @abstractmethod
    def test_any_generic(self): ...
    @abstractmethod
    def test_python_generic(self): ...
    @abstractmethod
    def test_self_generic(self): ...
