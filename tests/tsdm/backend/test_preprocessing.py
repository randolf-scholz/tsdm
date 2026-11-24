r"""Tests for tabular preprocessing helpers."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import tsdm.backend as B

type OutlierInput = (
    pa.Array
    | pa.ChunkedArray
    | pa.Table
    | pd.Series
    | pd.DataFrame
    | pl.Series
    | pl.DataFrame
)
type Expected = list[int | None] | dict[str, list[int | None]]
type OutlierMask = list[bool] | dict[str, list[bool]]
type OutlierBounds = Mapping[str, float | bool | None]

FIXED_DATA: dict[str, list[int | None]] = {
    "first": [-1, 0, 1, 2, 3, None],
    "second": [0, -3, 0, 3, 3, None],
}
SERIES_LIMITS: OutlierBounds = {
    "lower_bound": 0,
    "upper_bound": 2,
    "lower_inclusive": True,
    "upper_inclusive": True,
}
TABLE_LIMITS: dict[str, OutlierBounds] = {
    "first": SERIES_LIMITS,
    "second": {
        "lower_bound": -2,
        "upper_bound": 2,
        "lower_inclusive": True,
        "upper_inclusive": True,
    },
}
SERIES_EXPECTED: list[int | None] = [0, 1, 2, None]
TABLE_EXPECTED: dict[str, list[int | None]] = {
    "first": [None, 0, 1, 2, None],
    "second": [0, None, 0, None, None],
}
SERIES_MASK: list[bool] = [True, False, False, False, True, False]
TABLE_MASK: dict[str, list[bool]] = {
    "first": [True, False, False, False, True, False],
    "second": [False, True, False, True, True, False],
}
OUTLIER_CASES = [
    pytest.param(
        lambda: pa.chunked_array([FIXED_DATA["first"][:3], FIXED_DATA["first"][3:]]),
        SERIES_LIMITS,
        SERIES_EXPECTED,
        SERIES_MASK,
        pa.ChunkedArray,
        id="pyarrow-chunked-array",
    ),
    pytest.param(
        lambda: pa.array(FIXED_DATA["first"]),
        SERIES_LIMITS,
        SERIES_EXPECTED,
        SERIES_MASK,
        pa.Array,
        id="pyarrow-array",
    ),
    pytest.param(
        lambda: pa.table(FIXED_DATA),
        TABLE_LIMITS,
        TABLE_EXPECTED,
        TABLE_MASK,
        pa.Table,
        id="pyarrow-table",
    ),
    pytest.param(
        lambda: pd.Series(FIXED_DATA["first"], name="first"),
        SERIES_LIMITS,
        SERIES_EXPECTED,
        SERIES_MASK,
        pd.Series,
        id="pandas-series",
    ),
    pytest.param(
        lambda: pd.DataFrame(FIXED_DATA),
        TABLE_LIMITS,
        TABLE_EXPECTED,
        TABLE_MASK,
        pd.DataFrame,
        id="pandas-dataframe",
    ),
    pytest.param(
        lambda: pl.Series("first", FIXED_DATA["first"]),
        SERIES_LIMITS,
        SERIES_EXPECTED,
        SERIES_MASK,
        pl.Series,
        id="polars-series",
    ),
    pytest.param(
        lambda: pl.DataFrame(FIXED_DATA),
        TABLE_LIMITS,
        TABLE_EXPECTED,
        TABLE_MASK,
        pl.DataFrame,
        id="polars-dataframe",
    ),
]


def _normalize_values(values: Sequence[Any], /) -> list[int | None]:
    r"""Convert backend-specific null values to Python ``None``."""
    return [None if pd.isna(value) else int(value) for value in values]


def _normalize(result: OutlierInput, /) -> Expected:
    r"""Convert a supported result to a backend-independent representation."""
    match result:
        case pl.Series() as series:
            return _normalize_values(series.to_list())
        case pl.DataFrame() as frame:
            return {
                column: _normalize_values(values)
                for column, values in frame.to_dict(as_series=False).items()
            }
        case (pa.Array() | pa.ChunkedArray()) as array:
            return _normalize_values(array.to_pylist())
        case pa.Table() as table:
            return {
                column: _normalize_values(values)
                for column, values in table.to_pydict().items()
            }
        case pd.Series() as series:
            return _normalize_values(series.to_list())
        case pd.DataFrame() as frame:
            return {
                column: _normalize_values(frame[column].to_list())
                for column in frame.columns
            }

        case _:
            raise TypeError(f"Unsupported result type: {type(result)}")


def _normalize_mask(result: OutlierInput, /) -> OutlierMask:
    r"""Convert a supported boolean result to a Python representation."""
    match result:
        case pl.Series() as series:
            return [bool(value) for value in series.to_list()]
        case pl.DataFrame() as frame:
            return {
                column: [bool(value) for value in values]
                for column, values in frame.to_dict(as_series=False).items()
            }
        case (pa.Array() | pa.ChunkedArray()) as array:
            return [bool(value) for value in array.to_pylist()]
        case pa.Table() as table:
            return {
                column: [bool(value) for value in values]
                for column, values in table.to_pydict().items()
            }
        case pd.Series() as series:
            return [bool(value) for value in series.to_list()]
        case pd.DataFrame() as frame:
            return {
                column: [bool(value) for value in frame[column].to_list()]
                for column in frame.columns
            }
        case _:
            raise TypeError(f"Unsupported result type: {type(result)}")


@pytest.mark.parametrize(
    ("factory", "limits", "expected", "_mask", "result_type"), OUTLIER_CASES
)
@pytest.mark.parametrize("registered", [False, True])
def test_remove_outliers(
    factory: Callable[[], OutlierInput],
    limits: OutlierBounds | Mapping[str, OutlierBounds],
    expected: Expected,
    _mask: OutlierMask,
    result_type: type[OutlierInput],
    *,
    registered: bool,
) -> None:
    r"""Remove outliers consistently across all supported tabular backends."""
    obj = factory()
    backend = _backend(obj, registered=registered)
    result = backend.remove_outliers(obj, limits)

    assert isinstance(result, result_type)
    assert _normalize(result) == expected


@pytest.mark.parametrize(
    ("factory", "limits", "_expected", "mask", "result_type"), OUTLIER_CASES
)
@pytest.mark.parametrize("registered", [False, True])
def test_select_outliers(
    factory: Callable[[], OutlierInput],
    limits: OutlierBounds | Mapping[str, OutlierBounds],
    _expected: Expected,
    mask: OutlierMask,
    result_type: type[OutlierInput],
    *,
    registered: bool,
) -> None:
    r"""Select outliers consistently across all supported tabular backends."""
    obj = factory()
    backend = _backend(obj, registered=registered)
    result = backend.select_outliers(obj, limits)

    assert isinstance(result, result_type)
    assert _normalize_mask(result) == mask


def _backend(obj: OutlierInput, /, *, registered: bool) -> Any:
    r"""Choose the direct module or its registered kernel collection."""
    if registered:
        return B.get_backend(obj)
    match obj:
        case pa.Array() | pa.ChunkedArray() | pa.Table():
            return B.pyarrow
        case pd.Series() | pd.DataFrame():
            return B.pandas
        case pl.Series() | pl.DataFrame():
            return B.polars
        case _:
            raise TypeError(type(obj))


@pytest.mark.parametrize("backend", [B.pandas, B.polars, B.pyarrow])
def test_limits_formats(backend: Any) -> None:
    r"""Accept native limits tables and keyword bounds with equivalent results."""
    columns = {
        "variable": list(TABLE_LIMITS),
        "description": ["First variable", "Second variable"],
        **{
            key: [bounds[key] for bounds in TABLE_LIMITS.values()]
            for key in SERIES_LIMITS
        },
    }
    obj: Any
    limits: Any
    match backend:
        case B.pandas:
            obj = pd.DataFrame(FIXED_DATA)
            limits = pd.DataFrame(columns)
        case B.polars:
            obj = pl.DataFrame(FIXED_DATA)
            limits = pl.DataFrame(columns)
        case B.pyarrow:
            obj = pa.table(FIXED_DATA)
            limits = pa.table(columns)
        case _:
            raise TypeError(backend)
    keywords = {
        key: dict(zip(columns["variable"], columns[key], strict=True))
        for key in SERIES_LIMITS
    }
    assert _normalize(backend.remove_outliers(obj, limits)) == TABLE_EXPECTED
    assert _normalize(backend.remove_outliers(obj, **keywords)) == TABLE_EXPECTED
    assert _normalize(backend.remove_outliers(obj, keywords)) == TABLE_EXPECTED
    assert _normalize_mask(backend.select_outliers(obj, limits)) == TABLE_MASK
    assert _normalize_mask(backend.select_outliers(obj, **keywords)) == TABLE_MASK
    assert _normalize_mask(backend.select_outliers(obj, keywords)) == TABLE_MASK
    if isinstance(limits, pd.DataFrame):
        assert (
            _normalize(backend.remove_outliers(obj, limits.set_index("variable")))
            == TABLE_EXPECTED
        )
        assert (
            _normalize_mask(backend.select_outliers(obj, limits.set_index("variable")))
            == TABLE_MASK
        )
    with pytest.raises(ValueError, match="both"):
        backend.remove_outliers(obj, limits, **keywords)
    with pytest.raises(ValueError, match="Missing boundary"):
        backend.select_outliers(obj, lower_bound=0)


@pytest.mark.parametrize(
    ("backend", "factory", "foreign_factory"),
    [
        (backend, factory, foreign_factory)
        for backend, factory in [
            (B.pandas, pd.DataFrame),
            (B.polars, pl.DataFrame),
            (B.pyarrow, pa.table),
        ]
        for foreign_factory in [pd.DataFrame, pl.DataFrame, pa.table]
        if foreign_factory is not factory
    ],
)
@pytest.mark.parametrize("operation", ["select_outliers", "remove_outliers"])
def test_foreign_limits_table(
    backend: Any,
    factory: Callable[..., Any],
    foreign_factory: Callable[..., Any],
    *,
    operation: str,
) -> None:
    r"""Require explicit conversion of limits tables from a different backend."""
    limits = foreign_factory(
        {
            "variable": ["first"],
            **{key: [value] for key, value in SERIES_LIMITS.items()},
        }
    )
    obj = factory({"first": FIXED_DATA["first"]})
    with pytest.raises(TypeError, match="mapping of outlier limits"):
        getattr(backend, operation)(obj, limits)


@pytest.mark.parametrize("backend", [B.pandas, B.polars, B.pyarrow])
@pytest.mark.parametrize("registered", [False, True])
def test_strip_whitespace(backend: Any, *, registered: bool) -> None:
    r"""Trim selected or all string columns, retaining nulls and numeric values."""
    data = {"first": [" a ", None], "second": [" b ", " c "], "number": [1, 2]}
    obj: Any
    match backend:
        case B.pandas:
            obj = pd.DataFrame(data).astype({"first": "string", "second": "string"})
        case B.polars:
            obj = pl.DataFrame(data)
        case B.pyarrow:
            obj = pa.table(data)
        case _:
            raise TypeError(backend)
    kernel: Any = B.get_backend(obj) if registered else backend
    assert _string_values(kernel.strip_whitespace(obj, "first")) == {
        **data,
        "first": ["a", None],
    }
    assert _string_values(kernel.strip_whitespace(obj)) == {
        **data,
        "first": ["a", None],
        "second": ["b", "c"],
    }
    assert _string_values(obj) == data
    series = obj["first"]
    result = kernel.strip_whitespace(series)
    values = (
        result.to_pylist() if isinstance(result, pa.ChunkedArray) else result.to_list()
    )
    assert values[0] == "a"
    assert pd.isna(values[1])
    with pytest.raises(ValueError, match="columns"):
        kernel.strip_whitespace(series, "first")


def _string_values(frame: Any, /) -> dict[str, list[Any]]:
    r"""Normalize string tables, preserving nulls as `None`."""
    match frame:
        case pd.DataFrame():
            return (
                frame.astype(object).where(frame.notna(), None).to_dict(orient="list")
            )
        case pl.DataFrame():
            return frame.to_dict(as_series=False)
        case pa.Table():
            return frame.to_pydict()
        case _:
            raise TypeError(type(frame))
