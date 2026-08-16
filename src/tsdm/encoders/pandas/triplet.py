r"""Encode timeseries in triplet format."""

__all__ = ["TripletEncoder", "TripletDecoder"]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder
from tsdm.pprint import pprint_repr


@pprint_repr
@dataclass(init=False, repr=False)
class TripletEncoder(FittableEncoder[pd.DataFrame, pd.DataFrame]):
    r"""Converts wide DataFrame to a tall DataFrame.

    Requires that all columns share the same data type.
    If sparse, then
    """

    sparse: bool = False
    r"""Whether to use a sparse representation."""
    var_name: str = "variable"
    r"""The name of the variable column."""
    value_name: str = "value"
    r"""The name of the value column."""
    value_dtype: Any = UNDEFINED
    r"""The dtype of the variable column."""

    original_schema: pd.Series = UNDEFINED
    r"""The original schema (column -> dtype)."""
    categories: pd.CategoricalDtype = UNDEFINED
    r"""The stored categories."""

    def __init__(
        self,
        *,
        sparse: bool = False,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> None:
        self.sparse = sparse
        self.var_name = var_name
        self.value_name = value_name

    def fit(self, data: pd.DataFrame, /) -> None:
        self.original_schema = data.dtypes
        self.categories = pd.CategoricalDtype(data.columns)

        # check that all columns have the same dtype
        if len(variable_dtypes := set(data.dtypes)) != 1:
            raise ValueError("All columns must have the same dtype!")

        self.value_dtype = variable_dtypes.pop()

    def encode(self, data: pd.DataFrame, /) -> pd.DataFrame:
        df = (
            data.melt(
                ignore_index=False,
                var_name=self.var_name,
                value_name=self.value_name,
            )
            .dropna(how="any")
            .astype(
                {
                    self.var_name: self.categories,
                    self.value_name: self.value_dtype,
                }
            )
            .sort_index()
        )

        if self.sparse:
            df = pd.get_dummies(
                df,
                columns=[self.var_name],
                sparse=True,
                prefix="",
                prefix_sep="",
            )
            # move value column to the end
            return df[df.columns[1:].union(df.columns[:1])]

        return df

    def decode(self, data: pd.DataFrame, /) -> pd.DataFrame:
        if self.sparse:
            df = data.iloc[:, :-1].stack()
            df = df[df == 1]
            df.index = df.index.rename(self.var_name, level=-1)
            df = df.reset_index(level=-1)
            df[self.value_name] = data[self.value_name]
        else:
            df = data

        df = df.pivot_table(
            # FIXME: with https://github.com/pandas-dev/pandas/pull/45994
            # simply use df.index.names instead then.
            index=df.index,
            columns=self.var_name,
            values=self.value_name,
            dropna=False,
        )

        if isinstance(data.index, pd.MultiIndex):
            df.index = pd.MultiIndex.from_tuples(df.index, names=data.index.names)

        # re-add missing columns
        return df.reindex(columns=self.original_schema.index).astype(
            self.original_schema
        )


@pprint_repr
@dataclass(init=False, repr=False)
class TripletDecoder(FittableEncoder[pd.DataFrame, pd.DataFrame]):
    r"""Convert a tall DataFrame to a wide DataFrame."""

    sparse: bool = False
    r"""Whether to use a sparse representation."""
    value_name: str = UNDEFINED
    r"""The name of the value column."""
    var_name: str = UNDEFINED
    r"""The name of the variable column."""
    value_dtype: Any = UNDEFINED
    r"""The dtype of the variable column."""

    categories: pd.CategoricalDtype = UNDEFINED
    r"""The stored categories."""
    original_schema: Mapping[str, Any] = UNDEFINED
    r"""The original dtypes."""

    def __init__(
        self,
        *,
        sparse: bool = UNDEFINED,
        value_name: str = UNDEFINED,
        var_name: str = UNDEFINED,
        categories: pd.CategoricalDtype | Iterable = UNDEFINED,
    ) -> None:
        self.sparse = sparse
        self.var_name = var_name
        self.value_name = value_name
        self.categories = (
            pd.CategoricalDtype(categories)
            if isinstance(categories, Iterable)
            else categories
        )

    def fit(self, data: pd.DataFrame, /) -> None:
        if self.sparse is UNDEFINED:
            self.sparse = len(data.columns) > 2
        if self.var_name is UNDEFINED:
            self.var_name = "variable" if self.sparse else str(data.columns[0])
        if self.value_name is UNDEFINED:
            self.value_name = str(data.columns[-1])

        self.categories = (
            self.categories
            if self.categories is not UNDEFINED
            else pd.CategoricalDtype(data.columns[:-1])
            if self.sparse
            else pd.CategoricalDtype(data[self.var_name].unique())
        )

        self.value_dtype = data[self.value_name].dtype
        self.original_schema = data.dtypes.to_dict()

    def encode(self, data: pd.DataFrame, /) -> pd.DataFrame:
        if self.sparse:
            df = data.iloc[:, :-1].stack()
            df = df[df == 1]
            df.index = df.index.rename(self.var_name, level=-1)
            df = df.reset_index(level=-1)
            df[self.value_name] = data[self.value_name]
        else:
            df = data

        df = df.pivot_table(
            # TODO: FIX with https://github.com/pandas-dev/pandas/pull/45994
            # simply use df.index.names instead then.
            index=df.index,
            columns=self.var_name,
            values=self.value_name,
            dropna=False,
        )

        if isinstance(data.index, pd.MultiIndex):
            df.index = pd.MultiIndex.from_tuples(df.index, names=data.index.names)

        # re-add missing columns
        df = df.reindex(columns=self.categories.categories, fill_value=float("nan"))
        df.columns.name = self.var_name

        # Finalize result
        result = df[self.categories.categories]  # fix column order
        return result.sort_index()

    def decode(self, data: pd.DataFrame, /) -> pd.DataFrame:
        df = (
            data.melt(
                ignore_index=False,
                var_name=self.var_name,
                value_name=self.value_name,
            )
            .dropna(how="any")
            .astype(
                {
                    self.var_name: self.categories,
                    self.value_name: self.value_dtype,
                }
            )
            .sort_index()
        )

        if self.sparse:
            df = pd.get_dummies(
                df,
                columns=[self.var_name],
                sparse=True,
                prefix="",
                prefix_sep="",
            )
            # move value column to the end
            df = df[df.columns[1:].union(df.columns[:1])]

        return df.reindex(columns=self.original_schema).astype(self.original_schema)
