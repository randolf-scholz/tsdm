r"""Encoders for pandas DataFrames.

Each of these encoders must encode DataFrames.
We distinguish between 3 categories of encoders:

1. **polymodal encoders**: Accepts dataframes with different schemas, such as different number of columns.
2. **submodal encoders**: input dataframes must be subschemas of the schema used for fitting.
3. **equimodal encoders**: Each input dataframe must have the same schema as the dataframe used for fitting.
"""

__all__ = [
    # Classes
    "CSVEncoder",
    "DTypeConverter",
    "FrameAsDict",
    "FrameAsTensor",
    "FrameAsTensorDict",
    "FrameEncoder",
    "TripletDecoder",
    "TripletEncoder",
    # Functions
    "is_canonically_indexed",
    "get_ellipsis_cols",
]

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import KW_ONLY, asdict, dataclass
from pathlib import Path
from types import EllipsisType

import pandas as pd
import polars as pl
import pyarrow as pa
import torch
from pandas import DataFrame, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList
from torch import Tensor
from typing_extensions import Any, ClassVar, Optional, TypeVar

from tsdm.constants import EMPTY_MAP
from tsdm.encoders.base import BaseEncoder, Encoder
from tsdm.types.aliases import FilePath, PandasDtype, PandasDTypeArg
from tsdm.types.variables import K, T
from tsdm.utils.decorators import pprint_repr

E = TypeVar("E", bound=Encoder)
F = TypeVar("F", bound=Encoder)
TableVar = TypeVar("TableVar", DataFrame, pl.DataFrame, pa.Table)


def get_ellipsis_cols(
    df: DataFrame, /, schema: Iterable[EllipsisType | T | list[T]]
) -> list[T]:
    r"""Determine the column name for the ellipsis."""
    original_columns: set[T] = set(df.columns)
    selected_columns = original_columns.copy()

    if Ellipsis in original_columns:
        raise ValueError("Ellipsis is a reserved column name!")

    for el in schema:
        match el:
            case EllipsisType():
                continue
            case list() as cols:
                if original_columns.issuperset(cols):
                    selected_columns -= set(cols)
                    continue
                raise ValueError(f"Columns {cols} are not present in the DataFrame!")
            case col:
                if col in original_columns:
                    selected_columns.remove(col)
                    continue
                raise ValueError(f"Column {col} is not present in the DataFrame!")

    # NOTE: In order to get the columns in the original order,
    #  we need to iterate over the original columns.
    #  simply doing list(selected_columns) would yield the columns in a different order.
    return [col for col in df.columns if col in selected_columns]


def is_canonically_indexed(df: DataFrame, /) -> bool:
    r"""Check if the DataFrame has a canonical index."""
    match df.index:
        case pd.RangeIndex(start=0, step=1, stop=stop) if stop == len(df):  # type: ignore[has-type]
            return True
    return False


class FrameEncoder(BaseEncoder[DataFrame, DataFrame], Mapping[K, Encoder]):
    r"""Encode a DataFrame by group-wise transformations.

    Similar to `sklearn.compose.ColumnTransformer`.

    Per-column encoding is possible through the dictionary input.
    In this case, the positions of the columns in the encoded DataFrame should coincide with the
    positions of the columns in the input DataFrame.

    Todo: We want encoding groups, so for example, applying an encoder to a group of columns.

    - [ ] Add support for groups of column-encoders
    """

    original_columns: list[K]
    original_dtypes: Series
    original_index_columns: list[K]
    original_value_columns: list[K]

    encoders: Mapping[K, Encoder]
    column_encoders: Mapping[K, Encoder]
    index_encoders: Mapping[K, Encoder]

    @property
    def params(self) -> dict[str, Any]:
        return {
            "column_encoders": self.column_encoders,
            "index_encoders": self.index_encoders,
            "original_columns": self.original_columns,
            "original_dtypes": self.original_dtypes,
            "original_index_columns": self.original_index_columns,
            "original_value_columns": self.original_value_columns,
        }

    def __init__(
        self,
        column_encoders: Mapping[K, Encoder] = EMPTY_MAP,
        *,
        index_encoders: Mapping[K, Encoder] = EMPTY_MAP,
    ) -> None:
        self.index_encoders = index_encoders
        self.column_encoders = column_encoders
        self.encoders = {**column_encoders, **index_encoders}

    def __getitem__(self, key: K, /) -> Encoder:
        return self.encoders[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.encoders)

    def __len__(self) -> int:
        return len(self.encoders)

    def fit(self, data: DataFrame, /) -> None:
        data = data.copy(deep=True)
        index = data.index.to_frame()
        self.original_value_columns = FrozenList(data.columns)
        self.original_index_columns = FrozenList(index.columns)

        data = data.reset_index()
        self.original_dtypes = data.dtypes
        self.original_columns = FrozenList(data.columns)

        # fit the encoders one by one
        for group, encoder in self.encoders.items():
            try:
                encoder.fit(data[group])
            except Exception as exc:
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{group}]: Failed to fit {enc}.")
                raise

    def encode(self, data: DataFrame, /) -> DataFrame:
        data = data.reset_index()

        for group, encoder in self.encoders.items():
            data[group] = encoder.encode(data[group])

        index_columns = data.columns.intersection(self.original_index_columns)
        data = data.set_index(index_columns.tolist())
        return data

    def decode(self, data: DataFrame, /) -> DataFrame:
        data = data.reset_index()

        for group, encoder in self.encoders.items():
            data[group] = encoder.decode(data[group])

        # Restore index order + dtypes
        data = data.astype(self.original_dtypes[data.columns])
        index_columns = data.columns.intersection(self.original_index_columns)
        data = data.set_index(index_columns.tolist())
        return data


@pprint_repr
@dataclass(init=False, repr=False)
class TripletEncoder(BaseEncoder[DataFrame, DataFrame]):
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
    value_dtype: PandasDtype = NotImplemented
    r"""The dtype of the variable column."""

    original_schema: Series = NotImplemented
    r"""The original schema (column -> dtype)."""
    categories: pd.CategoricalDtype = NotImplemented
    r"""The stored categories."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

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

    def fit(self, data: DataFrame, /) -> None:
        self.original_schema = data.dtypes
        self.categories = pd.CategoricalDtype(data.columns)

        # check that all columns have the same dtype
        if len(variable_dtypes := set(data.dtypes)) != 1:
            raise ValueError("All columns must have the same dtype!")

        self.value_dtype = variable_dtypes.pop()

    def encode(self, data: DataFrame, /) -> DataFrame:
        df = (
            data.melt(
                ignore_index=False,
                var_name=self.var_name,
                value_name=self.value_name,
            )
            .dropna(how="any")
            .astype({
                self.var_name: self.categories,
                self.value_name: self.value_dtype,
            })
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

    def decode(self, data: DataFrame, /) -> DataFrame:
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

        if isinstance(data.index, MultiIndex):
            df.index = MultiIndex.from_tuples(df.index, names=data.index.names)

        # re-add missing columns
        return df.reindex(columns=self.original_schema.index).astype(
            self.original_schema
        )


@pprint_repr
@dataclass(init=False, repr=False)
class TripletDecoder(BaseEncoder[DataFrame, DataFrame]):
    r"""Convert a tall DataFrame to a wide DataFrame."""

    sparse: bool = False
    r"""Whether to use a sparse representation."""
    value_name: str = NotImplemented
    r"""The name of the value column."""
    var_name: str = NotImplemented
    r"""The name of the variable column."""
    value_dtype: PandasDtype = NotImplemented
    r"""The dtype of the variable column."""

    categories: pd.CategoricalDtype = NotImplemented
    r"""The stored categories."""
    original_schema: Series = NotImplemented
    r"""The original dtypes."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        *,
        sparse: bool = NotImplemented,
        value_name: str = NotImplemented,
        var_name: str = NotImplemented,
        categories: pd.CategoricalDtype | Sequence = NotImplemented,
    ) -> None:
        self.sparse = sparse
        self.var_name = var_name
        self.value_name = value_name

    def fit(self, data: DataFrame, /) -> None:
        if self.sparse is NotImplemented:
            self.sparse = len(data.columns) > 2
        if self.var_name is NotImplemented:
            self.var_name = "variable" if self.sparse else data.columns[0]
        if self.value_name is NotImplemented:
            self.value_name = data.columns[-1]

        self.categories = (
            pd.CategoricalDtype(data.columns[:-1])
            if self.sparse
            else pd.CategoricalDtype(data[self.var_name].unique())
        )

        self.value_dtype = data[self.value_name].dtype
        self.original_schema = data.dtypes

    def encode(self, data: DataFrame, /) -> DataFrame:
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

        if isinstance(data.index, MultiIndex):
            df.index = MultiIndex.from_tuples(df.index, names=data.index.names)

        # re-add missing columns
        df = df.reindex(columns=self.categories.categories, fill_value=float("nan"))
        df.columns.name = self.var_name

        # Finalize result
        result = df[self.categories.categories]  # fix column order
        return result.sort_index()

    def decode(self, data: DataFrame, /) -> DataFrame:
        df = (
            data.melt(
                ignore_index=False,
                var_name=self.var_name,
                value_name=self.value_name,
            )
            .dropna(how="any")
            .astype({
                self.var_name: self.categories,
                self.value_name: self.value_dtype,
            })
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

        return df.reindex(columns=self.original_schema.index).astype(
            self.original_schema
        )


@pprint_repr
@dataclass(init=False, slots=True)
class CSVEncoder(BaseEncoder[DataFrame, FilePath]):
    r"""Encode the data into a CSV file."""

    DEFAULT_READ_OPTIONS: ClassVar[dict] = {}
    DEFAULT_WRITE_OPTIONS: ClassVar[dict] = {"index": False}

    path_generator: Callable[[DataFrame], Path]
    r"""The generates the name for the CSV file."""

    _: KW_ONLY

    csv_read_options: dict[str, Any]
    r"""The options for the read_csv function."""
    csv_write_options: dict[str, Any]
    r"""The options for the to_csv function."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        filename_or_generator: FilePath | Callable[[DataFrame], Path],
        *,
        csv_write_options: Mapping[str, Any] = EMPTY_MAP,
        csv_read_options: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        self.csv_read_options = self.DEFAULT_READ_OPTIONS | dict(csv_read_options)
        self.csv_write_options = self.DEFAULT_WRITE_OPTIONS | dict(csv_write_options)
        self.path_generator = (
            filename_or_generator
            if callable(filename_or_generator)
            else lambda _: Path(filename_or_generator)
        )

    def encode(self, data: DataFrame, /) -> Path:
        path = self.path_generator(data)
        data.to_csv(path, **self.csv_write_options)
        return path

    def decode(self, str_or_path: FilePath, /) -> DataFrame:
        return pd.read_csv(str_or_path, **self.csv_read_options)


@pprint_repr
@dataclass(init=False, repr=False)
class DTypeConverter(BaseEncoder[DataFrame, DataFrame]):
    r"""Converts dtypes of a DataFrame.

    Note:
        - If a column is not present, it will be ignored.
        - If `...` (`Ellipsis`) is given,
          all unspecified columns will be converted to the given dtype.
          (Ellipsis will be removed during `.fit()`)
    """

    target_dtypes: dict[Any, PandasDTypeArg] = NotImplemented
    r"""The target dtypes."""
    original_dtypes: dict[str, Any] = NotImplemented
    r"""The original dtypes."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(self, dtypes: PandasDTypeArg | Mapping[Any, PandasDTypeArg]) -> None:
        super().__init__()
        self.target_dtypes = (
            dict(dtypes) if isinstance(dtypes, Mapping) else {...: dtypes}
        )

    def encode(self, data: DataFrame, /) -> DataFrame:
        return data.astype({k: self.target_dtypes[k] for k in data.columns})

    def decode(self, data: DataFrame, /) -> DataFrame:
        return data.astype({k: self.original_dtypes[k] for k in data.columns})

    def fit(self, data: DataFrame, /) -> None:
        self.original_dtypes = data.dtypes.to_dict()

        if Ellipsis in self.target_dtypes:
            fill_dtype = self.target_dtypes.pop(Ellipsis)
            ellipsis_cols: list = get_ellipsis_cols(data, self.target_dtypes)
            for col in ellipsis_cols:
                self.target_dtypes[col] = fill_dtype


@pprint_repr
@dataclass
class FrameAsTensor(BaseEncoder[DataFrame, Tensor]):
    r"""Converts a `DataFrame` to a `torch.Tensor`.

    Note:
        - This encoder requires that the DataFrame is canonically indexed. (i.e. `index = range(len(df))`)
        - This encoder requires that the DataFrame is of a single (numerical) dtype.
    """

    dtype: Optional[str | torch.dtype] = None
    r"""The default dtype."""
    device: Optional[str | torch.device] = None
    r"""The device the tensors are stored in."""
    original_schema: dict[str, Any] = NotImplemented
    r"""The original schema."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def fit(self, data: DataFrame, /) -> None:
        if data.index != pd.RangeIndex(len(data)):
            raise ValueError("DataFrame must be canonically indexed!")
        self.original_schema = data.dtypes.to_dict()

    def encode(self, data: DataFrame, /) -> Tensor:
        return torch.tensor(data.values, device=self.device, dtype=self.dtype)  # type: ignore[arg-type]

    def decode(self, data: Tensor, /) -> DataFrame:
        array = data.detach().cpu().numpy()
        frame = DataFrame(array, columns=self.original_schema)
        return frame.astype(self.original_schema)


@pprint_repr
@dataclass(init=False)
class FrameAsDict(BaseEncoder[DataFrame, dict[str, DataFrame]]):
    """Encodes a DataFrame as a dict of DataFrames.

    Note:
        - Each column must be assigned to exactly one group.
        - The special value `...` (`Ellipsis`) can be used to indicate that all unspecified columns belong to a group.
    """

    # Attributes (type hints represent post-fit attributes)
    schema: dict[str, list[str] | EllipsisType]
    r"""The schema for grouping the columns (group-name -> col-name(s))."""

    # Fitted attributes
    original_dtypes: dict[str, Any] = NotImplemented  # cols -> dtype

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        schema: Mapping[str, str | list[str] | EllipsisType],
    ) -> None:
        self.schema = {
            k: v if isinstance(v, list | EllipsisType) else [v]
            for k, v in schema.items()
        }

    def fit(self, data: DataFrame, /) -> None:
        # get the original dtypes
        self.original_dtypes = data.dtypes.to_dict()

        # fill in the missing columns
        if Ellipsis in self.schema.values():
            ellipsis_cols: list[str] = get_ellipsis_cols(data, self.schema.values())
            for group, cols in self.schema.items():
                if cols is Ellipsis:
                    self.schema[group] = ellipsis_cols

        if missing_cols := set().union(*self.schema.values()) - set(data.columns):  # type: ignore[arg-type]
            raise ValueError(f"Missing columns {missing_cols}!")
        if extra_cols := set(data.columns) - set().union(*self.schema.values()):  # type: ignore[arg-type]
            raise ValueError(f"Extra columns {extra_cols}!")

    def encode(self, data: DataFrame, /) -> dict[str, DataFrame]:
        r"""Encode a DataFrame as a dict of Tensors.

        The encode method ensures treatment of missingness:
        if columns in the dataframe are missing, the correponding tensor columns
        will be filled with `NAN`-values if the datatype allows it.
        """
        return {key: data[cols] for key, cols in self.schema.items()}

    def decode(self, data: Mapping[str, DataFrame], /) -> DataFrame:
        # Assemble the DataFrame
        return (
            pd.concat(data.values(), axis="columns")
            .astype(self.original_dtypes)  # restores dtypes
            .reindex(columns=self.original_dtypes)  # restores column order
        )


@pprint_repr
@dataclass(init=False)
class FrameAsTensorDict(BaseEncoder[DataFrame, dict[str, Tensor]]):
    r"""Encodes a DataFrame as a dict of Tensors.

    This is useful for passing a DataFrame to a PyTorch model.
    One can specify groups of columns to be encoded as a single Tensor.
    They must share the same dtype.

    Note:
        - Each column must be assigned to exactly one group.
        - The special value `...` (`Ellipsis`) can be used to indicate that all unspecified columns belong to a group.
        - Encoding the index is mandatory, except if the data is canonically indexed.
          (i.e. `index = RangeIndex(len(df))`)
        - This Encoder is basically equivalent to `FrameAsDict` followed by `MapEncoders` wrapping `FrameAsTensor`
          for each group, but it also encodes the index.
        - Missing columns are allowed (submodular), and will be filled with `NAN`-values if the datatype allows it.

    Example:
        >>> from pandas import DataFrame
        >>> from tsdm.encoders import FrameAsTensorDict
        >>> df = DataFrame({
        ...     "ID": [10, 21, 33],
        ...     "mask": [True, False, False],
        ...     "x": [-2.1, 7.3, 3.5],
        ...     "y": [0.1, 0.2, 0.3],
        ... }).set_index("ID")
        >>> encoder = FrameAsTensorDict({
        ...     "index": "ID",
        ...     "mask": "mask",
        ...     "features": ["x", "y"],
        ... })
        >>> encoder.fit(df)
        >>> encoded = encoder.encode(df)
        >>> assert isinstance(encoded, dict)
        >>> decoded = encoder.decode(encoded)
        >>> pd.testing.assert_frame_equal(df, decoded)
    """

    # Attributes (type hints represent post-fit attributes)
    schema: dict[str, list[str] | EllipsisType]
    r"""The schema for grouping the columns (group-name -> col-name(s))."""
    device: dict[str | EllipsisType, None | str | torch.device]
    r"""The device for each group (group-name -> device)."""
    dtypes: dict[str | EllipsisType, None | str | torch.dtype]
    r"""The dtype for each group (group-name -> dtype)."""

    # Fitted attributes
    index_cols: list[str] = NotImplemented
    original_dtypes: dict[str, Any] = NotImplemented  # cols -> dtype

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        schema: Mapping[str, str | list[str] | EllipsisType],
        *,
        device: Optional[
            str | torch.device | Mapping[str | EllipsisType, None | str | torch.device]
        ] = None,
        dtypes: Optional[
            str | torch.dtype | Mapping[str | EllipsisType, None | str | torch.dtype]
        ] = None,
    ) -> None:
        self.schema = {
            k: v if isinstance(v, list | EllipsisType) else [v]
            for k, v in schema.items()
        }
        self.dtypes = dict(dtypes) if isinstance(dtypes, Mapping) else {...: dtypes}
        self.device = dict(device) if isinstance(device, Mapping) else {...: device}

    def fit(self, data: DataFrame, /) -> None:
        # check the index of the dataframe
        self.index_cols = list(data.index.names)
        if not is_canonically_indexed(data):
            data = data.reset_index()

        # get the original dtypes
        self.original_dtypes = data.dtypes.to_dict()

        # fill in the missing columns
        if Ellipsis in self.schema.values():
            ellipsis_cols: list[str] = get_ellipsis_cols(data, self.schema.values())
            for group, cols in self.schema.items():
                if cols is Ellipsis:
                    self.schema[group] = ellipsis_cols

        # fill in the dtype for missing groups
        if Ellipsis in self.dtypes:
            dtype = self.dtypes.pop(Ellipsis)
            for group in self.schema.keys() - self.dtypes.keys():
                self.dtypes[group] = dtype

        # fill in the device for missing groups
        if Ellipsis in self.device:
            device = self.device.pop(Ellipsis)
            for group in self.schema.keys() - self.device.keys():
                self.device[group] = device

        if self.dtypes.keys() & self.device.keys() != self.schema.keys():
            raise ValueError(
                "Schema, dtypes and device columns must share groups!"
                f"\nSchema: {self.schema}"
                f"\ndtypes: {self.dtypes}"
                f"\ndevice: {self.device}"
            )

        if missing_cols := set().union(*self.schema.values()) - set(data.columns):  # type: ignore[arg-type]
            raise ValueError(f"Missing columns {missing_cols}!")
        if extra_cols := set(data.columns) - set().union(*self.schema.values()):  # type: ignore[arg-type]
            raise ValueError(f"Extra columns {extra_cols}!")

    def encode(self, data: DataFrame, /) -> dict[str, Tensor]:
        r"""Encode a DataFrame as a dict of Tensors.

        The encode method ensures treatment of missingness:
        if columns in the dataframe are missing, the correponding tensor columns
        will be filled with `NAN`-values if the datatype allows it.
        """
        data = data.reset_index()

        return {
            key: torch.tensor(
                data[cols].to_numpy(),
                device=self.device[key],
                dtype=self.dtypes[key],  # type: ignore[arg-type]
            ).squeeze()
            for key, cols in self.schema.items()
        }

    def decode(self, data: Mapping[str, Tensor], /) -> DataFrame:
        # convert the tensors to dataframes
        dfs = [
            DataFrame(tensor.detach().cpu().numpy(), columns=self.schema[key])
            for key, tensor in data.items()
        ]

        # Assemble the DataFrame
        df = (
            pd.concat(dfs, axis="columns")
            .astype(self.original_dtypes)  # restores dtypes
            .reindex(columns=self.original_dtypes)  # restores column order
        )

        if self.index_cols != [None]:
            df = df.set_index(self.index_cols)

        return df
