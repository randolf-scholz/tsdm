r"""Encoders for pandas DataFrames.

Each of these encoders must encode DataFrames.
We distinguish between 3 categories of encoders:

1. **polymodal encoders**: Accepts dataframes with different schemas, such as different number of columns.
2. **submodal encoders**: Each input dataframe must have a *compatible* schema as the dataframe used for fitting.
    - The columns must be a subset of the columns used for fitting.
    - The dtypes must be compatible.
3. **equimodal encoders**: Each input dataframe must have the same schema as the dataframe used for fitting.
"""

__all__ = [
    # Classes
    "CSVEncoder",
    "DTypeConverter",
    "FrameAsTensor",
    "FrameAsTensorDict",
    "FrameEncoder",
    "FrameSplitter",
    "TripletDecoder",
    "TripletEncoder",
]

import warnings
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass
from pathlib import Path
from types import EllipsisType

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList
from torch import Tensor
from typing_extensions import Any, ClassVar, Optional, TypeVar

from tsdm.constants import EMPTY_MAP
from tsdm.encoders.base import BaseEncoder, Encoder
from tsdm.types.aliases import FilePath, PandasDtype, PandasDTypeArg, PandasObject
from tsdm.types.dtypes import TORCH_DTYPES
from tsdm.types.protocols import NTuple
from tsdm.types.variables import K
from tsdm.utils import pairwise_disjoint
from tsdm.utils.decorators import pprint_repr

E = TypeVar("E", bound=Encoder)
F = TypeVar("F", bound=Encoder)
TableVar = TypeVar("TableVar", DataFrame, pl.DataFrame, pa.Table)


class FrameEncoder(BaseEncoder[DataFrame, DataFrame], Mapping[K, Encoder]):
    r"""Encode a DataFrame by group-wise transformations.

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

    Args:
        dtypes: A mapping from column names to dtypes.
            If a column is not present, it will be ignored.
            If `...` (`Ellipsis`) is given, all remaining columns will be converted to the given dtype.
            When decoding, the original dtypes will be restored.
    """

    original_dtypes: Series = NotImplemented
    r"""The original dtypes."""
    target_dtypes: dict[Any, PandasDTypeArg] = NotImplemented
    r"""The target dtypes."""
    fill_dtype: Optional[PandasDtype] = None
    r"""The dtype to fill missing columns with."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self, dtypes: PandasDTypeArg | Mapping[Any, PandasDTypeArg], /
    ) -> None:
        super().__init__()
        match dtypes:
            case Mapping() as mapping:
                self.target_dtypes = dict(mapping)
            case dtype:
                self.target_dtypes = {Ellipsis: dtype}

        if Ellipsis in self.target_dtypes:
            self.fill_dtype = self.target_dtypes[Ellipsis]

    def encode(self, data: DataFrame, /) -> DataFrame:
        if Ellipsis in self.target_dtypes:
            return data.astype({
                k: self.target_dtypes.get(k, self.fill_dtype) for k in data.columns
            })
        return data.astype({k: self.target_dtypes[k] for k in data.columns})

    def decode(self, data: DataFrame, /) -> DataFrame:
        return data.astype({k: self.original_dtypes[k] for k in data.columns})

    def fit(self, data: DataFrame, /) -> None:
        self.original_dtypes = data.dtypes.copy()

        if Ellipsis in self.target_dtypes:
            if Ellipsis in data.columns:
                raise ValueError("Ellipsis is a reserved column name!")

            self.fill_dtype = self.target_dtypes.pop(Ellipsis)
            for col in set(data.columns) - set(self.target_dtypes):
                self.target_dtypes[col] = self.fill_dtype


class FrameSplitter(BaseEncoder[DataFrame, tuple[DataFrame, ...]], Mapping):
    r"""Splits a DataFrame into multiple DataFrames column-wise.

    Note:
        - The mapping must be one-to-one, that is each column must be assigned to exactly one group.
        - The special value `...` (`Ellipsis`) can be used to indicate that all unspecified columns belong to a group.

    Example:
        The index mapping `[0|1|2|3|4|5]` to `[2|0|1], [5|4]` corresponds to mapping

        +---+---+---+---+---+---+
        | 0 | 1 | 2 | 3 | 4 | 5 |
        +===+===+===+===+===+===+
        | 1 | 2 | 0 | - | 5 | 4 |
        +---+---+---+---+---+---+

        with inverse

        +---+---+---+---+---+---+
        | 0 | 1 | 2 | 3 | 4 | 5 |
        +===+===+===+===+===+===+
        | 1 | 2 | 0 | - | 5 | 4 |
        +---+---+---+---+---+---+
    """

    original_columns: Index
    original_dtypes: Series

    groups: dict[Any, EllipsisType | Hashable | list[Hashable]]
    group_indices: dict[Any, list[int]]

    has_ellipsis: bool = False
    ellipsis_columns: Optional[list[Hashable]] = None
    ellipsis: Optional[Hashable] = None

    permutation: list[int]
    inverse_permutation: list[int]
    rtype: type = tuple

    def __init__(
        self,
        groups: Iterable[Hashable] | Mapping[Any, Hashable],
        /,
        *,
        dropna: bool = False,
        fillna: bool = True,
    ) -> None:
        if isinstance(groups, NTuple):
            self.rtype = type(groups)
            groups = groups._asdict()
        if not isinstance(groups, Mapping):
            groups = dict(enumerate(groups))

        self.groups = {}
        for key, obj in groups.items():
            match obj:
                case EllipsisType():
                    self.groups[key] = obj
                    self.ellipsis = key
                    self.has_ellipsis = True
                case str() as name:
                    self.groups[key] = [name]
                case Iterable() as iterable:
                    self.groups[key] = list(iterable)
                case _:
                    self.groups[key] = [obj]

        column_sets: list[set[Hashable]] = [
            set(cols) for cols in self.groups.values() if isinstance(cols, Iterable)
        ]
        self.fixed_columns = set().union(*column_sets)
        assert pairwise_disjoint(column_sets)

        self.dropna = dropna
        self.fillna = fillna

    def __len__(self) -> int:
        r"""Return the number of groups."""
        return len(self.groups)

    def __iter__(self) -> Iterator:
        r"""Iterate over the groups."""
        return iter(self.groups)

    def __getitem__(self, item: Any, /) -> Hashable | list[Hashable]:
        r"""Return the group."""
        return self.groups[item]

    def fit(self, original: DataFrame, /) -> None:
        data = DataFrame(original).copy()

        self.original_dtypes = original.dtypes
        self.original_columns = original.columns

        if self.has_ellipsis:
            self.ellipsis_columns = [
                c for c in data.columns if c not in self.fixed_columns
            ]
        else:
            unused_columns = (
                set() if self.has_ellipsis else set(data.columns) - self.fixed_columns
            )
            data = data.drop(columns=unused_columns)

        columns_index = data.columns.to_series().reset_index(drop=True)
        reverse_index = Series(columns_index.index, index=columns_index)

        # Compute the permutation
        self.permutation = []
        self.group_indices: dict[Any, list[int]] = {}
        for group, columns in self.groups.items():
            if columns is Ellipsis:
                self.group_indices[group] = reverse_index[
                    self.ellipsis_columns
                ].to_list()
            else:
                self.group_indices[group] = reverse_index[columns].to_list()
            self.permutation += self.group_indices[group]

        # compute inverse permutation
        self.inverse_permutation = np.argsort(self.permutation).tolist()
        # self.inverse_permutation sorted(p.copy(), key=p.__getitem__)

    def encode(self, original: DataFrame, /) -> tuple[DataFrame, ...]:
        data = DataFrame(original).copy()

        if not self.has_ellipsis and set(data.columns) > self.fixed_columns:
            warnings.warn(
                f"Unknown columns {set(data.columns) - self.fixed_columns}."
                "If you want to encode unknown columns add a group `...` (`Ellipsis`).",
                RuntimeWarning,
                stacklevel=2,
            )

        encoded_frames = []
        for columns in self.groups.values():
            cols = self.ellipsis_columns if columns is Ellipsis else columns
            encoded = data[cols]
            if self.dropna:
                encoded = encoded.dropna(axis="index", how="all")
            encoded_frames.append(encoded)

        return tuple(encoded_frames)

    def decode(self, data: tuple[DataFrame, ...], /) -> DataFrame:
        data = tuple(DataFrame(x) for x in data)
        joined = pd.concat(data, axis="columns")

        # bring columns in order
        joined = joined.iloc[..., self.inverse_permutation]
        reconstructed = DataFrame(columns=self.original_columns)
        reconstructed[joined.columns] = joined
        reconstructed = reconstructed.astype(self.original_dtypes)

        if self.dropna:
            reconstructed = reconstructed.sort_index()
        return reconstructed


@pprint_repr
@dataclass
class FrameAsTensor(BaseEncoder[PandasObject, Tensor]):
    r"""Converts a `DataFrame` to a `torch.Tensor`.

    Note:
        - This encoder requires that the DataFrame is canonically indexed. (i.e. `index = range(len(df))`)
        - This encoder requires that the DataFrame is of a single (numerical) dtype.
    """

    dtype: Optional[torch.dtype] = None
    r"""The default dtype."""
    device: Optional[torch.device] = None
    r"""The device the tensors are stored in."""
    original_schema: Series = NotImplemented
    r"""The original schema."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def fit(self, data: PandasObject, /) -> None:
        if data.index != pd.RangeIndex(len(data)):
            raise ValueError("DataFrame must be canonically indexed!")
        self.original_schema = data.dtypes

    def encode(self, data: PandasObject, /) -> Tensor:
        return torch.tensor(data.values, device=self.device, dtype=self.dtype)

    def decode(self, data: Tensor, /) -> PandasObject:
        array = data.cpu().numpy()
        frame = DataFrame(array, columns=self.original_schema.index)
        return frame.astype(self.original_schema)


class FrameAsTensorDict(
    BaseEncoder[DataFrame, dict[str, Tensor]], Mapping[str, list[str]]
):
    r"""Encodes a DataFrame as a dict of Tensors.

    This is useful for passing a DataFrame to a PyTorch model.
    One can specify groups of columns to be encoded as a single Tensor. They must share the same dtype.

    Example:
        >>> from pandas import DataFrame
        >>> from tsdm.encoders import FrameAsTensorDict
        >>> df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        >>> encoder = FrameAsTensorDict(
        ...     groups={"a": ["a", "b"], "c": ["c"]}, encode_index=False
        ... )
        >>> encoder.fit(df)
        >>> encoded = encoder.encode(df)
        >>> assert isinstance(encoded, dict)
        >>> decoded = encoder.decode(encoded)
        >>> pd.testing.assert_frame_equal(df, decoded)
    """

    # Attributes
    original_index_columns: Index | list[str]
    original_schema: Series = NotImplemented
    inferred_dtypes: dict[str, torch.dtype | None]
    groups: dict[str, list[str]]

    # Parameters
    column_dtype: Optional[torch.dtype] = None
    device: Optional[str | torch.device] = None
    dtypes: dict[str, None | str | torch.dtype]
    encode_index: Optional[bool] = None
    index_dtype: Optional[torch.dtype] = None

    def __init__(
        self,
        *,
        groups: dict[str, list[str] | EllipsisType],
        dtypes: Optional[dict[str, str]] = None,
        device: Optional[str | torch.device | Mapping[str, str | torch.dtype]] = None,
        encode_index: Optional[bool] = None,
    ) -> None:
        self.groups = groups  # type: ignore[assignment]
        self.dtypes = dtypes  # type: ignore[assignment]
        self.device = device  # type: ignore[assignment]
        self.encode_index = encode_index
        self.inferred_dtypes = {}

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[str]:
        return iter(self.groups)

    def __getitem__(self, key: str, /) -> list[str]:
        return self.groups[key]

    def fit(self, data: DataFrame, /) -> None:
        index = data.index.to_frame()
        self.original_index_columns = index.columns.to_list()

        # if None try to autodetect if index-columns are being used
        if self.encode_index is None:
            self.encode_index = bool(
                set()
                .union(*(v for v in self.groups.values() if v is not Ellipsis))
                .intersection(self.original_index_columns)
            )

        if self.encode_index:
            data = data.reset_index()

        if len(data.columns) != len(data.columns.unique()):
            raise ValueError("Duplicate columns currently not supported!")

        self.original_dtypes = data.dtypes
        self.original_columns = data.columns

        # Impute Ellipsis. Make sure that the order is preserved.
        cols: set[str] = set(data.columns)
        ellipsis_key: Optional[str] = None
        for key in self.groups:
            if self.groups[key] is Ellipsis:
                ellipsis_key = key
            else:
                s = set(self.groups[key])
                if not s.issubset(cols):
                    if s.issubset(cols | set(self.original_index_columns)):
                        raise ValueError(
                            "Index columns are not allowed in groups. "
                            "Please use encode_index=True."
                        )
                    raise ValueError(
                        f"{s} is not a subset of {cols}! Maybe you have a typo?"
                    )
                cols -= s
        if ellipsis_key is not None:
            self.groups[ellipsis_key] = [col for col in data.columns if col in cols]
            cols -= cols

        if cols:
            raise ValueError(
                f"Columns {cols} are not assigned to a group! "
                "Try setting encode_index=False to skip encoding the index."
            )

        # data type validation
        for key in self.groups:
            if data[self.groups[key]].dtypes.nunique() != 1:
                raise ValueError("All members of a group must have the same dtype!")

        # dtype validation
        if not isinstance(self.dtypes, Mapping):
            self.dtypes = dict.fromkeys(self.groups, self.dtypes)  # type: ignore[unreachable]
        for key in self.groups:
            val = self.dtypes.get(key, None)
            assert isinstance(val, None | str | torch.dtype)
            self.dtypes[key] = val
            self.inferred_dtypes[key] = (
                TORCH_DTYPES[val] if isinstance(val, str) else val
            )

    def encode(self, data: DataFrame, /) -> dict[str, Tensor]:
        r"""Encode a DataFrame as a dict of Tensors.

        The encode method ensures treatment of missingness:
        if columns in the dataframe are missing, the correponding tensor columns
        will be filled with `NAN`-values if the datatype allows it.
        """
        if self.encode_index:
            data = data.reset_index()

        return {
            key: torch.tensor(
                data[cols].astype(self.dtypes[key]).to_numpy(),
                device=self.device,
                dtype=self.inferred_dtypes[key],
            ).squeeze()
            for key, cols in self.groups.items()
            if set(cols).issubset(data.columns)
        }

    def decode(self, data: Mapping[str, Tensor | None], /) -> DataFrame:
        dfs = []
        for key, tensor in data.items():
            cols = self.groups[key]
            dtypes = self.original_dtypes[cols]
            df = DataFrame(
                [] if tensor is None else tensor.clone().detach().cpu(),
                columns=cols,
            ).astype(dtypes)
            dfs.append(df)

        # Assemble the DataFrame
        df = pd.concat(dfs, axis="columns")
        df = df.astype(self.original_dtypes[df.columns])
        df = df[self.original_columns.intersection(df.columns)]
        if self.encode_index and self.original_index_columns is not None:
            cols = [col for col in self.original_index_columns if col in df.columns]
            df = df.set_index(cols)
        return df
