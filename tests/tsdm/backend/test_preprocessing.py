r"""Tests for tabular preprocessing helpers."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import tsdm.backend as B

type Table = pa.Table | pd.DataFrame | pl.DataFrame
type TableFactory = Callable[..., Table]
type Tabular = Table | pa.Array | pa.ChunkedArray | pd.Series | pl.Series
type Values[T] = list[T] | dict[str, list[T]]
type OutlierBounds = Mapping[str, float | bool | None]


class TestCase(NamedTuple):
    factory: Callable[[], Tabular]
    limits: OutlierBounds | Mapping[str, OutlierBounds]
    expected_values: Values[int | None]
    expected_mask: Values[bool]
    mask_type: type[Tabular] | None = None


TABLE_FACTORIES: dict[str, TableFactory] = {
    "pandas": pd.DataFrame,
    "polars": pl.DataFrame,
    "pyarrow": pa.table,
}


def _values(obj: Tabular, /) -> Values[Any]:
    r"""Convert backend values to Python values, using `None` for nulls."""
    match obj:
        case pl.Series():
            return obj.to_list()
        case pl.DataFrame():
            return obj.to_dict(as_series=False)
        case pa.Array() | pa.ChunkedArray():
            return obj.to_pylist()
        case pa.Table():
            return obj.to_pydict()
        case pd.Series():
            return obj.astype(object).where(obj.notna(), None).to_list()
        case pd.DataFrame():
            return obj.astype(object).where(obj.notna(), None).to_dict(orient="list")
        case _:
            raise TypeError(f"Unsupported result type: {type(obj)}")


def _backend(obj: Tabular, /, *, registered: bool = False) -> Any:
    r"""Choose the direct module or its registered kernel collection."""
    if registered:
        return B.get_backend(obj)
    match obj:
        case pl.Series() | pl.DataFrame():
            return B.polars
        case pd.Series() | pd.DataFrame():
            return B.pandas
        case pa.Array() | pa.ChunkedArray() | pa.Table():
            return B.pyarrow
        case _:
            raise TypeError(f"Unsupported backend for {type(obj)}")


class TestOutliers:
    SERIES_DATA = [-1, 0, 1, 2, 3, None]
    TABLE_DATA = {
        "first": SERIES_DATA,
        "second": [0, -3, 0, 3, 3, None],
    }
    KEYWORD_LIMITS: dict[str, dict[str, float | bool | None]] = {
        "lower_bound": {"first": 0, "second": -2},
        "upper_bound": {"first": 2, "second": 2},
        "lower_inclusive": {"first": True, "second": True},
        "upper_inclusive": {"first": True, "second": True},
    }
    SERIES_LIMITS: OutlierBounds = {
        bound: columns["first"] for bound, columns in KEYWORD_LIMITS.items()
    }
    TABLE_LIMITS: dict[str, OutlierBounds] = {
        "first": SERIES_LIMITS,
        "second": {
            bound: columns["second"] for bound, columns in KEYWORD_LIMITS.items()
        },
    }
    LIMITS_COLUMNS = {
        "variable": list(TABLE_DATA),
        "description": ["First variable", "Second variable"],
        **{bound: list(columns.values()) for bound, columns in KEYWORD_LIMITS.items()},
    }
    SERIES_EXPECTED = [0, 1, 2, None]
    TABLE_EXPECTED = {
        "first": [None, 0, 1, 2, None],
        "second": [0, None, 0, None, None],
    }
    SERIES_MASK = [True, False, False, False, True, False]
    TABLE_MASK = {
        "first": SERIES_MASK,
        "second": [False, True, False, True, True, False],
    }
    CASES: dict[str, TestCase] = {
        "pyarrow-chunked-array": TestCase(
            factory=partial(pa.chunked_array, [SERIES_DATA[:3], SERIES_DATA[3:]]),
            limits=SERIES_LIMITS,
            expected_values=SERIES_EXPECTED,
            expected_mask=SERIES_MASK,
            mask_type=pa.BooleanArray,
        ),
        "pyarrow-array": TestCase(
            factory=partial(pa.array, SERIES_DATA),
            limits=SERIES_LIMITS,
            expected_values=SERIES_EXPECTED,
            expected_mask=SERIES_MASK,
            mask_type=pa.BooleanArray,
        ),
        "pyarrow-table": TestCase(
            factory=partial(pa.table, TABLE_DATA),
            limits=TABLE_LIMITS,
            expected_values=TABLE_EXPECTED,
            expected_mask=TABLE_MASK,
        ),
        "pandas-series": TestCase(
            factory=partial(pd.Series, SERIES_DATA, name="first"),
            limits=SERIES_LIMITS,
            expected_values=SERIES_EXPECTED,
            expected_mask=SERIES_MASK,
        ),
        "pandas-dataframe": TestCase(
            factory=partial(pd.DataFrame, TABLE_DATA),
            limits=TABLE_LIMITS,
            expected_values=TABLE_EXPECTED,
            expected_mask=TABLE_MASK,
        ),
        "polars-series": TestCase(
            factory=partial(pl.Series, "first", SERIES_DATA),
            limits=SERIES_LIMITS,
            expected_values=SERIES_EXPECTED,
            expected_mask=SERIES_MASK,
        ),
        "polars-dataframe": TestCase(
            factory=partial(pl.DataFrame, TABLE_DATA),
            limits=TABLE_LIMITS,
            expected_values=TABLE_EXPECTED,
            expected_mask=TABLE_MASK,
        ),
    }
    FOREIGN_TABLE_FACTORIES = {
        f"{name}-with-{foreign_name}": (factory, foreign_factory)
        for name, factory in TABLE_FACTORIES.items()
        for foreign_name, foreign_factory in TABLE_FACTORIES.items()
        if name != foreign_name
    }

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("registered", [False, True])
    def test_remove_outliers(self, case: str, *, registered: bool) -> None:
        r"""Remove outliers and preserve the input container type."""
        test_case = self.CASES[case]
        obj = test_case.factory()
        backend = _backend(obj, registered=registered)
        result = backend.remove_outliers(obj, test_case.limits)

        assert isinstance(result, type(obj))
        assert _values(result) == test_case.expected_values

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("registered", [False, True])
    def test_select_outliers(self, case: str, *, registered: bool) -> None:
        r"""Select outliers with the backend's boolean mask container."""
        test_case = self.CASES[case]
        obj = test_case.factory()
        backend = _backend(obj, registered=registered)
        result = backend.select_outliers(obj, test_case.limits)

        assert isinstance(result, test_case.mask_type or type(obj))
        assert _values(result) == test_case.expected_mask

    @pytest.mark.parametrize("case", TABLE_FACTORIES)
    def test_limits_formats(self, case: str) -> None:
        r"""Accept native limits tables, bound mappings, and keyword bounds."""
        factory = TABLE_FACTORIES[case]
        obj = factory(self.TABLE_DATA)
        limits = factory(self.LIMITS_COLUMNS)
        backend = _backend(obj)
        remove = backend.remove_outliers
        select = backend.select_outliers

        assert _values(remove(obj, limits)) == self.TABLE_EXPECTED
        assert _values(remove(obj, self.KEYWORD_LIMITS)) == self.TABLE_EXPECTED
        assert _values(remove(obj, **self.KEYWORD_LIMITS)) == self.TABLE_EXPECTED
        assert _values(select(obj, limits)) == self.TABLE_MASK
        assert _values(select(obj, self.KEYWORD_LIMITS)) == self.TABLE_MASK
        assert _values(select(obj, **self.KEYWORD_LIMITS)) == self.TABLE_MASK

    def test_indexed_pandas_limits(self) -> None:
        r"""Use the pandas limits index to identify variables."""
        obj = pd.DataFrame(self.TABLE_DATA)
        limits = pd.DataFrame(self.LIMITS_COLUMNS).set_index("variable")

        assert _values(B.pandas.remove_outliers(obj, limits)) == self.TABLE_EXPECTED
        assert _values(B.pandas.select_outliers(obj, limits)) == self.TABLE_MASK

    @pytest.mark.parametrize("case", TABLE_FACTORIES)
    def test_conflicting_limits(self, case: str) -> None:
        r"""Reject positional limits combined with keyword bounds."""
        factory = TABLE_FACTORIES[case]
        obj = factory(self.TABLE_DATA)
        limits = factory(self.LIMITS_COLUMNS)

        with pytest.raises(ValueError, match="both"):
            _backend(obj).remove_outliers(obj, limits, **self.KEYWORD_LIMITS)

    @pytest.mark.parametrize("case", TABLE_FACTORIES)
    @pytest.mark.parametrize("operation", ["select_outliers", "remove_outliers"])
    def test_missing_boundary_mappings(self, case: str, *, operation: str) -> None:
        r"""Reject table calls with incomplete boundary keywords."""
        factory = TABLE_FACTORIES[case]
        obj = factory(self.TABLE_DATA)
        kernel = getattr(_backend(obj), operation)

        with pytest.raises(TypeError):
            kernel(obj, lower_bound=0)

    @pytest.mark.parametrize("case", FOREIGN_TABLE_FACTORIES)
    @pytest.mark.parametrize("operation", ["select_outliers", "remove_outliers"])
    def test_foreign_limits_table(self, case: str, *, operation: str) -> None:
        r"""Require explicit conversion of limits tables from a different backend."""
        factory, foreign_factory = self.FOREIGN_TABLE_FACTORIES[case]
        obj = factory(self.TABLE_DATA)
        limits = foreign_factory(self.LIMITS_COLUMNS)

        with pytest.raises(TypeError, match="mapping of outlier limits"):
            getattr(_backend(obj), operation)(obj, limits)


class TestStripWhitespace:
    DATA = {"first": [" a ", None], "second": [" b ", " c "], "number": [1, 2]}
    PANDAS_DTYPES = {"first": "string", "second": "string"}
    FIRST_TRIMMED = ["a", None]
    SELECTED_TRIMMED = {**DATA, "first": FIRST_TRIMMED}
    ALL_TRIMMED = {**SELECTED_TRIMMED, "second": ["b", "c"]}

    @pytest.mark.parametrize("case", TABLE_FACTORIES)
    @pytest.mark.parametrize("registered", [False, True])
    def test_strip_whitespace(self, case: str, *, registered: bool) -> None:
        r"""Trim selected or all string columns, retaining nulls and numeric values."""
        factory = TABLE_FACTORIES[case]
        obj = (
            pd.DataFrame(self.DATA).astype(self.PANDAS_DTYPES)
            if factory is pd.DataFrame
            else factory(self.DATA)
        )
        kernel = _backend(obj, registered=registered)

        assert _values(kernel.strip_whitespace(obj, "first")) == self.SELECTED_TRIMMED
        assert _values(kernel.strip_whitespace(obj)) == self.ALL_TRIMMED
        assert _values(obj) == self.DATA

        series = obj["first"]
        assert _values(kernel.strip_whitespace(series)) == self.FIRST_TRIMMED
        with pytest.raises(ValueError, match="columns"):
            kernel.strip_whitespace(series, "first")
