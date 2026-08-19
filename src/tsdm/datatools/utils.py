r"""Generic data utilities."""

__all__ = [
    "data_overview",
    "date_range",
    "describe",
    "get_dtypes",
    "get_schema",
    "timedelta",
    "timedelta_range",
    "timestamp",
    "validate_schema",
]

import datetime as dt
from collections.abc import Mapping
from functools import wraps
from os import PathLike, fspath
from typing import Any, Optional

import pandas as pd
import polars as pl
import pyarrow as pa
from pandas import Timedelta, Timestamp
from pandas._libs import NaTType
from scipy import stats

from tsdm.types.aliases import FilePath, FileStream
from tsdm.types.extra import (
    SupportsArray,
    SupportsDataFrame,
    SupportsDtype,
)


def get_schema(
    table: pd.MultiIndex
    | pd.Index
    | pd.Series
    | pd.DataFrame
    | pa.Table
    | pl.DataFrame,
    /,
) -> tuple[list[Any], dict[Any, Any], list[Any]]:
    r"""Get the columns, dtypes, and index columns of a table-like object.

    Returns:
        A tuple containing the column names, a mapping of column names to dtypes,
        and the index column names.

    Raises:
        NotImplementedError: If the table type is not supported.
    """
    match table:
        case pd.MultiIndex(names=names, dtypes=dtypes):
            return names, dict(zip(names, dtypes, strict=True)), []
        case pd.Index() as index:
            return [index.name], {index.name: index.dtype}, []
        case pd.Series() as series:
            return [series.name], {series.name: series.dtype}, series.index.names
        case pd.DataFrame() as df:
            return df.columns.tolist(), df.dtypes.to_dict(), df.index.names
        case pa.Table(schema=schema):
            return schema.names, dict(zip(schema.names, schema.types, strict=True)), []
        case pl.DataFrame() as df:
            return df.columns, {col: df[col].dtype for col in df.columns}, []
        case _:
            raise NotImplementedError(f"Cannot get schema for {type(table)} objects!")


def get_dtypes(table: object) -> list[object]:
    match table:
        # DataFrame-like
        case pd.DataFrame(dtypes=dtypes) | pd.MultiIndex(dtypes=dtypes):
            return list(dtypes)
        case pa.Table(schema=schema):
            return list(schema.types)
        case pl.DataFrame(dtypes=dtypes):
            return list(dtypes)
        case SupportsDataFrame() as supports_frame:
            frame: pd.DataFrame = supports_frame.__dataframe__()
            return list(frame.dtypes)
        # Tensor-like
        case pa.Array(type=dtype):
            return [dtype]
        case SupportsDtype(dtype=dtype):
            return [dtype]
        case SupportsArray() as array:
            return [array.__array__().dtype]
        case _:
            raise TypeError(f"Unsupported object type {type(table)}.")


def describe(
    s: pd.Series | pd.DataFrame,
    /,
    *,
    quantiles: tuple[float, ...] = (0, 0.01, 0.5, 0.99, 1),
) -> pd.DataFrame:
    r"""Describe a DataFrame on a per column basis."""
    if isinstance(s, pd.DataFrame):
        df = s
        return pd.concat([describe(df[col]) for col in df])

    # crete a zero-scalar of the same type as the series
    num = len(s)

    # compute the special values
    max_value = s.max()
    min_value = s.min()
    mode_value = s.mode().iloc[0]

    try:
        mean_value = s.mean()
        median_value = s.median()
        std_value = s.std()
    except Exception:
        mean_value = float("nan")
        std_value = float("nan")
        median_value = float("nan")

    # compute counts
    max_count = (s == max_value).sum()
    min_count = (s == min_value).sum()
    mode_count = (s == mode_value).sum()
    nan_count = s.isna().sum()
    unique_count = s.nunique()

    try:
        idx = s.first_valid_index()
        # NOTE: dropna is necessary for duplicate index
        val = s.loc[idx].dropna().iloc[0] if idx is not None else 0
        zero = val - val
        neg_count = (s < zero).sum()
        pos_count = (s > zero).sum()
        zero_count = (s == zero).sum()
    except Exception:
        neg_count = pd.NA
        pos_count = pd.NA
        zero_count = pd.NA

    # compute the rates
    max_rate = max_count / num
    min_rate = min_count / num
    mode_rate = mode_count / num
    nan_rate = nan_count / num
    neg_rate = neg_count / num
    pos_rate = pos_count / num
    unique_rate = unique_count / num
    zero_rate = zero_count / num

    # stats
    try:
        entropy = stats.entropy(s.value_counts(), base=2)
    except Exception:
        entropy = pd.NA
    try:
        quantile_values = s.quantile(quantiles)
    except Exception:
        quantile_values = pd.Series([float("nan")] * len(quantiles))

    return pd.DataFrame(
        {
            # stats
            ("stats", "entropy"): entropy,
            **{
                ("quantile", f"{q:.2f}"): v
                for q, v in zip(quantiles, quantile_values, strict=True)
            },
            # special values
            ("value", "max"): max_value,
            ("value", "mean"): mean_value,
            ("value", "median"): median_value,
            ("value", "min"): min_value,
            ("value", "mode"): mode_value,
            ("value", "std"): std_value,
            # counts
            ("count", "max"): max_count,
            ("count", "min"): min_count,
            ("count", "mode"): mode_count,
            ("count", "nan"): nan_count,
            ("count", "neg"): neg_count,
            ("count", "pos"): pos_count,
            ("count", "unique"): unique_count,
            ("count", "zero"): zero_count,
            # rates
            ("rate", "max"): max_rate,
            ("rate", "min"): min_rate,
            ("rate", "mode"): mode_rate,
            ("rate", "nan"): nan_rate,
            ("rate", "neg"): neg_rate,
            ("rate", "pos"): pos_rate,
            ("rate", "unique"): unique_rate,
            ("rate", "zero"): zero_rate,
        },
        index=[s.name],
    )


def data_overview(
    df: pd.DataFrame, /, *, index_col: Optional[int | str] = None, digits: int = 2
) -> pd.DataFrame:
    r"""Get a summary of the data."""
    overview = pd.DataFrame(index=df.columns)
    null_values = df.isna()
    numerical_cols = df.select_dtypes(include="number").columns

    overview["datapoints"] = (~null_values).sum()
    overview["num_unique"] = df.nunique()
    overview["missing"] = (null_values.mean() * 100).round(2)

    overview.loc[numerical_cols, "min"] = df[numerical_cols].min()
    overview.loc[numerical_cols, "mean"] = df[numerical_cols].mean()
    overview.loc[numerical_cols, "std"] = df[numerical_cols].std()
    overview.loc[numerical_cols, "max"] = df[numerical_cols].max()

    column_dtypes = {
        "datapoints" : "int64[pyarrow]",
        "num_unique" : "int64[pyarrow]",
        "missing"    : "float64[pyarrow]",
        "min"        : "float64[pyarrow]",
        "mean"       : "float64[pyarrow]",
        "std"        : "float64[pyarrow]",
        "max"        : "float64[pyarrow]",
    }  # fmt: skip

    overview = overview.astype(column_dtypes)
    for col, dtype in column_dtypes.items():
        if pd.api.types.is_float_dtype(dtype):
            overview[col] = overview[col].round(digits)

    if index_col is not None:
        freq = {}
        for col in df.columns:
            mask = pd.notna(df[col].squeeze())
            time = df.index.get_level_values(index_col)[mask]
            freq[col] = pd.Series(time).diff().mean()
        overview["freq"] = pd.Series(freq)
    return overview


def validate_schema(
    file: FilePath | FileStream,
    schema: Mapping[str, Any],
    /,
    *,
    separator: str = ",",
) -> None:
    r"""Validate that a CSV header exactly matches a schema.

    Args:
        file: CSV path or seekable file-like object.
        schema: Expected column names and data types, in CSV order.
        separator: CSV field separator.

    Raises:
        ValueError: If the file is not seekable or its header does not match the
            schema's columns and order.
    """
    match file:
        case str() | PathLike():
            actual_columns = pl.read_csv(
                fspath(file),
                has_header=True,
                infer_schema=False,
                n_rows=0,
                separator=separator,
            ).columns
        case stream:
            if not stream.seekable():
                raise ValueError(
                    "Cannot validate the header of a non-seekable CSV stream."
                )

            position = stream.tell()
            try:
                actual_columns = pl.read_csv(
                    stream,
                    has_header=True,
                    infer_schema=False,
                    n_rows=0,
                    separator=separator,
                ).columns
            finally:
                stream.seek(position)
                assert stream.tell() == position

    expected_columns = list(schema)
    missing_cols = set(actual_columns) - set(expected_columns)
    superfluous_cols = set(expected_columns) - set(actual_columns)
    if missing_cols or superfluous_cols:
        raise ValueError(
            "CSV header does not match schema: "
            f"\n\t    missing: {sorted(missing_cols)!r}"
            f"\n\tsuperfluous: {sorted(superfluous_cols)!r}"
        )
    assert len(actual_columns) == len(expected_columns)

    if actual_columns != expected_columns:
        raise ValueError(
            "CSV header and schema columns are in different orders: "
            f"\n\texpected: {expected_columns!r}"
            f"\n\t  actual: {actual_columns!r}"
        )


@wraps(Timedelta)
def timedelta(value: Any = ..., unit: Optional[str] = None, **kwargs: Any) -> Timedelta:
    r"""Utility function that ensures that the constructor does not return NaT."""
    td = (
        Timedelta(unit=unit, **kwargs)
        if value is Ellipsis
        else Timedelta(value, unit=unit, **kwargs)
    )
    if isinstance(td, NaTType):
        raise TypeError("Constructor returned NaT")
    return td


@wraps(Timestamp)
def timestamp(value: Any = ..., **kwargs: Any) -> Timestamp:
    r"""Utility function that ensures that the constructor does not return NaT."""
    ts = Timestamp(**kwargs) if value is Ellipsis else Timestamp(value, **kwargs)
    if isinstance(ts, NaTType):
        raise TypeError("Constructor returned NaT")
    return ts


def date_range(
    start: str | dt.datetime,
    stop: str | dt.datetime,
    *,
    freq: str | dt.timedelta,
    include_end: bool = True,
) -> list[dt.datetime]:
    t0 = timestamp(start)
    t1 = timestamp(stop)
    f = timedelta(freq)
    k = (t1 - t0) // f
    items = [t0 + i * f for i in range(k)]

    if include_end and items[-1] < t1:
        items.append(t1)
    return items


def timedelta_range(
    start: str | dt.timedelta,
    stop: str | dt.timedelta,
    *,
    freq: str | dt.timedelta,
    include_end: bool = True,
) -> list[dt.timedelta]:
    t0 = timedelta(start)
    t1 = timedelta(stop)
    f = timedelta(freq)
    k = (t1 - t0) // f
    items = [t0 + i * f for i in range(k)]

    if include_end and items[-1] < t1:
        items.append(t1)
    return items
