r"""Implementation of encoders."""

__all__ = [
    # Classes
    "FrameAsDict",
    "FrameAsTuple",
    "FrameEncoder",
    "FastFrameEncoder",
    "FrameIndexer",
    "FrameSplitter",
    "TensorEncoder",
    "TripletDecoder",
    "TripletEncoder",
    "ValueEncoder",
]

import warnings
from collections import namedtuple
from collections.abc import Hashable, Iterable, Iterator, Mapping
from types import EllipsisType
from typing import Any, ClassVar, Generic, Optional, TypeVar, cast, overload

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList
from torch import Tensor
from typing_extensions import deprecated

from tsdm.constants import EMPTY_MAP
from tsdm.encoders.base import BaseEncoder
from tsdm.types.aliases import PandasObject, PathLike
from tsdm.types.dtypes import TORCH_DTYPES
from tsdm.types.protocols import NTuple
from tsdm.types.variables import key_var as K
from tsdm.utils import pairwise_disjoint
from tsdm.utils.strings import repr_mapping

ColEncVar = TypeVar("ColEncVar", bound=BaseEncoder | Mapping[Any, BaseEncoder])
IndEncVar = TypeVar("IndEncVar", bound=BaseEncoder | Mapping[Any, BaseEncoder])


class CSVEncoder(BaseEncoder):
    r"""Encode the data into a CSV file."""

    requires_fit: ClassVar[bool] = True

    filename: PathLike
    r"""The filename of the CSV file."""
    dtypes: Series
    r"""The original dtypes."""
    read_csv_kwargs: dict[str, Any]
    r"""The kwargs for the read_csv function."""
    to_csv_kwargs: dict[str, Any]
    r"""The kwargs for the to_csv function."""

    def __init__(
        self,
        filename: PathLike,
        *,
        to_csv_kwargs: Optional[dict[str, Any]] = None,
        read_csv_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.read_csv_kwargs = read_csv_kwargs or {}
        self.to_csv_kwargs = to_csv_kwargs or {}

    def fit(self, data: DataFrame, /) -> None:
        self.dtypes = data.dtypes

    def encode(self, data: DataFrame, /) -> PathLike:
        data.to_csv(self.filename, **self.to_csv_kwargs)
        return self.filename

    def decode(self, data: Optional[PathLike] = None, /) -> DataFrame:
        if data is None:
            data = self.filename
        frame = pd.read_csv(data, **self.read_csv_kwargs)
        return DataFrame(frame).astype(self.dtypes)


@deprecated("deprecated in favor of encoders.FastFrameEncoder")
class FrameEncoder(BaseEncoder, Generic[ColEncVar, IndEncVar]):
    r"""Encode a DataFrame by group-wise transformations.

    Per-column encoding is possible through the dictionary input.
    In this case, the positions of the columns in the encoded DataFrame should coincide with the
    positions of the columns in the input DataFrame.

    Todo: We want encoding groups, so for example, applying an encoder to a group of columns.

    - [ ] Add support for groups of column-encoders
    """

    requires_fit: ClassVar[bool] = True

    columns: Index
    dtypes: Series
    index_columns: Index
    index_dtypes: Series
    duplicate: bool = False

    column_encoders: ColEncVar
    r"""Encoders for the columns."""
    column_decoders: ColEncVar
    r"""Reverse Dictionary from encoded column name -> encoder"""
    index_encoders: IndEncVar
    r"""Optional Encoder for the index."""
    index_decoders: IndEncVar
    r"""Reverse Dictionary from encoded index name -> encoder"""

    @staticmethod
    def _names(obj: Index | Series | DataFrame, /) -> Hashable | FrozenList[Hashable]:
        if isinstance(obj, MultiIndex):
            return FrozenList(obj.names)
        if isinstance(obj, Series | Index):
            return obj.name
        if isinstance(obj, DataFrame):
            return FrozenList(obj.columns)
        raise ValueError

    def __init__(
        self,
        column_encoders: Optional[ColEncVar] = None,
        *,
        index_encoders: Optional[IndEncVar] = None,
        duplicate: bool = False,
    ):
        super().__init__()

        if column_encoders is None:
            self.column_encoders = cast(ColEncVar, None)
        else:
            self.column_encoders = column_encoders

        if index_encoders is None:
            self.index_encoders = cast(IndEncVar, None)
        else:
            self.index_encoders = index_encoders

        # self.column_encoders = column_encoders
        # self.index_encoders = index_encoders

        self.duplicate = duplicate

    def fit(self, data: DataFrame, /) -> None:
        data = data.copy()
        index = data.index.to_frame()
        self.columns = data.columns
        self.dtypes = data.dtypes
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        # if self.duplicate:
        #     if not isinstance(self.column_encoders, BaseEncoder):
        #         raise ValueError("Duplication only allowed when single encoder")
        #     self.column_encoders = {
        #         col: deepcopy(self.column_encoders) for col in data.columns
        #     }

        if self.column_encoders is None:
            self.column_decoders = self.column_encoders  # type: ignore[unreachable]
        elif isinstance(self.column_encoders, BaseEncoder):
            self.column_encoders.fit(data)
            self.column_decoders = cast(ColEncVar, self.column_encoders)
        elif isinstance(self.column_encoders, Mapping):
            self.column_decoders = cast(ColEncVar, {})
            for group, encoder in self.column_encoders.items():
                encoder.fit(data[group])
                encoded = encoder.encode(data[group])
                self.column_decoders[self._names(encoded)] = encoder  # type: ignore[index]
        else:
            raise TypeError(f"Invalid {type(self.column_encoders)=}")

        if self.index_encoders is None:
            self.index_decoders = self.index_encoders  # type: ignore[unreachable]
        elif isinstance(self.index_encoders, BaseEncoder):
            self.index_encoders.fit(index)
            self.index_decoders = cast(IndEncVar, self.index_encoders)
        elif isinstance(self.index_encoders, Mapping):
            self.index_decoders = cast(IndEncVar, {})
            for group, encoder in self.index_encoders.items():
                encoder.fit(index[group])
                encoded = encoder.encode(index[group])
                self.index_decoders[self._names(encoded)] = encoder  # type: ignore[index]
        else:
            raise TypeError(f"Invalid {type(self.index_encoders)=}")

    def encode(self, data: DataFrame, /) -> DataFrame:
        data = data.copy(deep=True)
        index = data.index.to_frame()
        encoded_cols = data
        encoded_inds = encoded_cols.index.to_frame()

        if self.column_encoders is None:
            pass
        elif isinstance(self.column_encoders, BaseEncoder):
            encoded = self.column_encoders.encode(data)
            encoded_cols = encoded_cols.drop(columns=data.columns)
            encoded_cols = pd.concat([encoded_cols, encoded], axis="columns")
        elif isinstance(self.column_encoders, Mapping):
            for group, encoder in self.column_encoders.items():
                encoded = encoder.encode(data[group])
                encoded_cols = encoded_cols.drop(columns=group)
                # encoded_cols.loc[:,self._names(encoded)] = encoded
                encoded_cols = pd.concat([encoded_cols, encoded], axis="columns")
        else:
            raise TypeError(f"Invalid {type(self.column_encoders)=}")

        if self.index_encoders is None:
            pass
        elif isinstance(self.index_encoders, BaseEncoder):
            encoded = self.index_encoders.encode(index)
            encoded_inds = encoded_inds.drop(columns=index.columns)
            # encoded_inds.loc[:, self._names(encoded)] = encoded
            encoded_inds = pd.concat([encoded_inds, encoded], axis="columns")
        elif isinstance(self.index_encoders, Mapping):
            for group, encoder in self.index_encoders.items():
                encoded = encoder.encode(index[group])
                encoded_inds = encoded_inds.drop(columns=group)
                # encoded_inds.loc[:, self._names(encoded)] = encoded
                encoded_inds = pd.concat([encoded_inds, encoded], axis="columns")
        else:
            raise TypeError(f"Invalid {type(self.index_encoders)=}")

        # # Assemble DataFrame
        # encoded = encoded_cols.join(encoded_inds)  # DataFrame(encoded_cols)
        # # encoded.loc[:, self._names(encoded_inds)] = encoded_inds
        # encoded = encoded.set_index(self._names(encoded_inds))

        if isinstance(encoded_inds, Series):
            encoded = encoded_cols.set_index(encoded_inds)
        else:
            new_index = MultiIndex.from_frame(encoded_inds)
            encoded = encoded_cols.set_index(new_index)
        encoded = encoded[self.columns]
        return encoded

    def decode(self, data: DataFrame, /) -> DataFrame:
        data = data.copy(deep=True)
        index = data.index.to_frame()
        decoded_cols = data
        decoded_inds = decoded_cols.index.to_frame()

        if self.column_decoders is None:
            pass
        elif isinstance(self.column_decoders, BaseEncoder):
            decoded = self.column_decoders.decode(data)
            decoded_cols = decoded_cols.drop(columns=data.columns)
            decoded_cols[self._names(decoded)] = decoded
        elif isinstance(self.column_decoders, Mapping):
            for group, encoder in self.column_decoders.items():
                decoded = encoder.decode(data[group])
                decoded_cols = decoded_cols.drop(columns=group)
                names = self._names(decoded)
                decoded_cols[names] = decoded
        else:
            raise TypeError(f"Invalid {type(self.column_decoders)=}")

        if self.index_decoders is None:
            pass
        elif isinstance(self.index_decoders, BaseEncoder):
            decoded = self.index_decoders.decode(index)
            decoded_inds = decoded_inds.drop(columns=index.columns)
            decoded_inds[self._names(decoded)] = decoded
        elif isinstance(self.index_decoders, Mapping):
            for group, encoder in self.index_decoders.items():
                decoded = encoder.decode(index[group])
                decoded_inds = decoded_inds.drop(columns=group)
                decoded_inds[self._names(decoded)] = decoded
        else:
            raise TypeError(f"Invalid {type(self.index_decoders)=}")

        # Restore index order + dtypes
        decoded_inds = decoded_inds[
            self.index_columns.intersection(decoded_inds.columns)
        ]
        decoded_inds = decoded_inds.astype(self.index_dtypes[decoded_inds.columns])

        # Assemble DataFrame
        decoded = DataFrame(decoded_cols)
        decoded[self._names(decoded_inds)] = decoded_inds
        decoded = decoded.set_index(self._names(decoded_inds))
        decoded = decoded[self.columns]
        decoded = decoded.astype(self.dtypes)

        return decoded

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        items = {
            "column_encoders": self.column_encoders,
            "index_encoders": self.index_encoders,
        }
        return repr_mapping(items, title=self.__class__.__name__, recursive=2)


class FastFrameEncoder(BaseEncoder, Mapping[K, BaseEncoder]):
    r"""Encode a DataFrame by group-wise transformations.

    Per-column encoding is possible through the dictionary input.
    In this case, the positions of the columns in the encoded DataFrame should coincide with the
    positions of the columns in the input DataFrame.

    Todo: We want encoding groups, so for example, applying an encoder to a group of columns.

    - [ ] Add support for groups of column-encoders
    """

    requires_fit: ClassVar[bool] = True

    original_columns: list[K]
    original_dtypes: Series
    original_index_columns: list[K]
    original_value_columns: list[K]

    encoders: Mapping[K, BaseEncoder]
    column_encoders: Mapping[K, BaseEncoder]
    index_encoders: Mapping[K, BaseEncoder]

    def __init__(
        self,
        column_encoders: Mapping[K, BaseEncoder] = EMPTY_MAP,
        *,
        index_encoders: Mapping[K, BaseEncoder] = EMPTY_MAP,
    ) -> None:
        super().__init__()
        self.index_encoders = index_encoders
        self.column_encoders = column_encoders
        self.encoders = {**column_encoders, **index_encoders}

    def __getitem__(self, key: K) -> BaseEncoder:
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
                raise RuntimeError(
                    f"Failed to fit {type(encoder)} on {group=}"
                ) from exc

    def encode(self, data: DataFrame, /) -> DataFrame:
        data = data.reset_index()

        for group, encoder in self.encoders.items():
            data[group] = encoder.encode(data[group])

        index_columns = data.columns.intersection(self.original_index_columns)
        data = data.set_index(index_columns.tolist())
        return data

    def decode(self, data: DataFrame, /) -> DataFrame:
        data = data.reset_index()
        # index = data.index.to_frame()

        for group, encoder in self.encoders.items():
            data[group] = encoder.decode(data[group])

        # Restore index order + dtypes
        data = data.astype(self.original_dtypes[data.columns])
        index_columns = data.columns.intersection(self.original_index_columns)
        data = data.set_index(index_columns.tolist())
        return data


class FrameIndexer(BaseEncoder):
    r"""Change index of a `pandas.DataFrame`.

    For compatibility, this is done by integer index.
    """

    requires_fit: ClassVar[bool] = True

    index_columns: Index
    index_dtypes: Series
    index_indices: list[int]
    reset: EllipsisType | Hashable | list[Hashable]

    def __init__(self, *, reset: Optional[Hashable | list[Hashable]] = None) -> None:
        super().__init__()
        if reset is None:
            self.reset = []
        elif reset is Ellipsis:
            self.reset = Ellipsis
        elif isinstance(reset, str | int | tuple):
            self.reset = [reset]
        elif isinstance(reset, Iterable):
            self.reset = list(reset)
        else:
            raise TypeError("levels must be None, str, int, tuple or Iterable")

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"{self.__class__.__name__}(levels={self.reset})"

    def fit(self, data: DataFrame, /) -> None:
        index = data.index.to_frame()
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        if self.reset is Ellipsis or not isinstance(self.reset, list):
            self.index_indices = list(range(len(index.columns)))
        else:
            self.index_indices = list(range(len(self.reset)))

    def encode(self, data: DataFrame, /) -> DataFrame:
        return data.reset_index(level=self.reset)

    def decode(self, data: DataFrame, /) -> DataFrame:
        data = DataFrame(data)
        columns = data.columns[self.index_indices].to_list()
        return data.set_index(columns)


class FrameSplitter(BaseEncoder, Mapping):
    r"""Split DataFrame columns into multiple groups.

    The special value `...` (`Ellipsis`) can be used to indicate that all other columns belong to this group.
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

    requires_fit: ClassVar[bool] = True

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
        super().__init__()

        if isinstance(groups, NTuple):
            self.rtype = type(groups)
            groups = groups._asdict()
        if not isinstance(groups, Mapping):
            groups = dict(enumerate(groups))

        self.groups = {}
        for key, obj in groups.items():
            if obj is Ellipsis:
                self.groups[key] = obj
                self.ellipsis = key
                self.has_ellipsis = True
            elif isinstance(obj, str) or not isinstance(obj, Iterable):
                self.groups[key] = [obj]
            else:
                self.groups[key] = list(obj)

        column_sets: list[set[Hashable]] = [
            set(cols) for cols in self.groups.values() if isinstance(cols, Iterable)
        ]
        self.fixed_columns = set().union(*column_sets)
        assert pairwise_disjoint(column_sets)

        # self.keep_index = keep_index
        self.dropna = dropna
        self.fillna = fillna

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return repr_mapping(self)

    def __len__(self) -> int:
        r"""Return the number of groups."""
        return len(self.groups)

    def __iter__(self) -> Iterator:
        r"""Iterate over the groups."""
        return iter(self.groups)

    def __getitem__(self, item: Any) -> Hashable | list[Hashable]:
        r"""Return the group."""
        return self.groups[item]

    def fit(self, original: DataFrame, /) -> None:
        data = DataFrame(original).copy()

        # if self.dropna and not data.index.is_monotonic_increasing:
        #     raise ValueError(f"If {self.dropna=}, Index must be monotonic increasing!")

        self.original_dtypes = original.dtypes
        self.original_columns = original.columns

        # self.variable_indices = {col: [] for col in self.original_columns}
        # for group, columns in self.groups.items():
        #     if columns is Ellipsis:
        #         continue
        #     for column in columns:
        #         self.variable_indices[column].append(group)
        # self.inverse_groups = {}
        # for group, columns in self.groups.items():
        #     if columns is Ellipsis:
        #         continue
        #     for column in columns:
        #         self.inverse_groups[column] = group

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
            if columns is Ellipsis:
                encoded = data[self.ellipsis_columns]
            else:
                encoded = data[columns]
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


class TripletEncoder(BaseEncoder):
    r"""Encode the data into triplets."""

    requires_fit: ClassVar[bool] = True

    categories: pd.CategoricalDtype
    r"""The stored categories."""
    original_dtypes: Series
    r"""The original dtypes."""
    original_columns: Index
    r"""The original columns."""

    def __init__(
        self,
        *,
        sparse: bool = False,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.var_name = var_name
        self.value_name = value_name

    def fit(self, data: DataFrame, /) -> None:
        self.categories = pd.CategoricalDtype(data.columns)
        self.original_dtypes = data.dtypes
        self.original_columns = data.columns

    def encode(self, data: DataFrame, /) -> DataFrame:
        result = data.melt(
            ignore_index=False,
            var_name=self.var_name,
            value_name=self.value_name,
        ).dropna()

        result[self.var_name] = result[self.var_name].astype(self.categories)

        if self.sparse:
            result = pd.get_dummies(
                result, columns=[self.var_name], sparse=True, prefix="", prefix_sep=""
            )

        result = result.sort_index()
        return result

    def decode(self, data: DataFrame, /) -> DataFrame:
        if self.sparse:
            df = data.iloc[:, 1:].stack()
            df = df[df == 1]
            df.index = df.index.rename(self.var_name, level=-1)
            df = df.reset_index(level=-1)
            df[self.value_name] = data["self.value_name"]
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

        # Finalize result
        result = df[self.categories.categories]  # fix column order
        result = result.astype(self.original_dtypes)
        result = result[self.original_columns]
        result.columns = self.original_columns
        return result


class TripletDecoder(BaseEncoder):
    r"""Encode the data into triplets."""

    requires_fit: ClassVar[bool] = True

    categories: pd.CategoricalDtype
    r"""The stored categories."""
    original_dtypes: Series
    r"""The original dtypes."""
    original_columns: Index
    r"""The original columns."""
    value_column: Hashable
    r"""The name of the value column."""
    channel_columns: Index | Hashable
    r"""The name of the channel column(s)."""

    def __init__(
        self,
        *,
        sparse: bool = False,
        var_name: Optional[str] = None,
        value_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.var_name = var_name
        self.value_name = value_name

    def fit(self, data: DataFrame, /) -> None:
        self.original_dtypes = data.dtypes
        self.original_columns = data.columns

        self.value_column = self.value_name or data.columns[0]
        self.value_name = self.value_column
        assert self.value_column in data.columns

        remaining_cols = data.columns.drop(self.value_column)
        if self.sparse and len(remaining_cols) <= 1:
            raise ValueError("Sparse encoding requires at least two channel columns.")
        if not self.sparse and len(remaining_cols) != 1:
            raise ValueError("Dense encoding requires exactly one channel column.")

        if self.sparse:
            self.channel_columns = remaining_cols
            categories = self.channel_columns
            self.var_name = self.channel_columns.name or "variable"
        else:
            assert len(remaining_cols) == 1
            self.channel_columns = remaining_cols.item()
            categories = data[self.channel_columns].unique()
            self.var_name = self.channel_columns

        if pd.api.types.is_float_dtype(categories):
            raise ValueError(
                f"channel_ids found in {self.var_name!r} does no look like a"
                " categorical!\n Please specify `value_name` and/or `var_name`!"
            )

        self.categories = pd.CategoricalDtype(np.sort(categories))

    def encode(self, data: DataFrame, /) -> DataFrame:
        if self.sparse:
            df = data.loc[:, self.channel_columns].stack()
            df = df[df == 1]
            df.index = df.index.rename(self.var_name, level=-1)
            df = df.reset_index(level=-1)
            df[self.value_name] = data[self.value_column]
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
        result = data.melt(
            ignore_index=False,
            var_name=self.var_name,
            value_name=self.value_name,
        ).dropna()

        if self.sparse:
            result = pd.get_dummies(
                result, columns=[self.var_name], sparse=True, prefix="", prefix_sep=""
            )

        result = result.astype(self.original_dtypes)
        result = result.sort_index()
        return result


class TensorEncoder(BaseEncoder):
    r"""Converts objects to Tensor."""

    requires_fit: ClassVar[bool] = True

    dtype: torch.dtype
    r"""The default dtype."""
    device: torch.device
    r"""The device the tensors are stored in."""
    names: Optional[list[str]] = None
    return_type: type = tuple

    def __init__(
        self,
        *,
        names: Optional[list[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.names = names
        self.dtype = torch.float32 if dtype is None else dtype
        # default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") if device is None else device

        if names is not None:
            self.return_type = namedtuple("namedtuple", names)  # type: ignore[misc]

        self.is_fitted = True

    def fit(self, data: PandasObject, /) -> None:
        pass

    @overload
    def encode(self, data: PandasObject, /) -> Tensor: ...  # type: ignore[misc]
    @overload
    def encode(self, data: tuple[PandasObject, ...], /) -> tuple[Tensor, ...]: ...
    def encode(self, data, /):
        if isinstance(data, tuple):
            return tuple(self.encode(x) for x in data)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
        if isinstance(data, Index | Series | DataFrame):
            return torch.tensor(data.values, device=self.device, dtype=self.dtype)
        return torch.tensor(data, device=self.device, dtype=self.dtype)

    @overload
    def decode(self, data: Tensor, /) -> PandasObject: ...
    @overload
    def decode(self, data: tuple[Tensor, ...], /) -> tuple[PandasObject, ...]: ...
    def decode(self, data, /):
        if isinstance(data, tuple):
            return tuple(self.decode(x) for x in data)
        return data.cpu().numpy()


class ValueEncoder(BaseEncoder):
    r"""Encodes the value of a DataFrame.

    Remembers dtypes, index, columns
    """

    requires_fit: ClassVar[bool] = True

    index_columns: Index
    index_dtypes: Series
    column_columns: Index
    column_dtypes: Series
    original_columns: Index
    original_dtypes: Series
    dtype: Optional[str] = None

    def __init__(self, dtype: Optional[str] = None, /) -> None:
        super().__init__()
        self.dtype = dtype

    def fit(self, data: DataFrame, /) -> None:
        index = data.index.to_frame()
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes
        self.column_columns = data.columns
        self.column_dtypes = data.dtypes
        self.original_columns = data.reset_index().columns
        self.original_dtypes = data.dtypes

        if self.original_dtypes.nunique() != 1 and self.dtype is None:
            warnings.warn(
                "Non-uniform dtype detected!"
                "This may cause unexpected behavior."
                "Please specify dtype.",
                RuntimeWarning,
                stacklevel=2,
            )

    def encode(self, data: DataFrame, /) -> NDArray:
        array = data.reset_index().values
        return array.astype(self.dtype)

    def decode(self, data: NDArray, /) -> DataFrame:
        frame = DataFrame(data, columns=self.original_columns)

        # Assemble the columns
        columns = frame[self.column_columns]
        columns.columns = self.column_columns
        columns = columns.astype(self.column_dtypes)
        columns = columns.squeeze(axis="columns")

        # assemble the index
        index = frame[self.index_columns]
        index.columns = self.index_columns
        index = index.astype(self.index_dtypes)
        index = index.squeeze(axis="columns")

        if isinstance(index, Series):
            decoded = columns.set_index(index)
        else:
            decoded = columns.set_index(MultiIndex.from_frame(index))
        return decoded


class FrameAsDict(BaseEncoder, Mapping[str, list[str]]):
    r"""Encodes a DataFrame as a dict of Tensors.

    This is useful for passing a DataFrame to a PyTorch model.
    One can specify groups of columns to be encoded as a single Tensor. They must share the same dtype.

    .. code-block:: pycon

        >>> from pandas import DataFrame
        >>> from tsdm.encoders import FrameAsDict
        >>> df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        >>> encoder = FrameAsDict(groups={"a": ["a", "b"], "c": ["c"]}, encode_index=False)
        >>> encoder.fit(df)
        >>> encoded = encoder.encode(df)
        >>> assert isinstance(encoded, dict)
        >>> decoded = encoder.decode(encoded)
        >>> pd.testing.assert_frame_equal(df, decoded)
    """

    requires_fit: ClassVar[bool] = True

    # Attributes
    original_index_columns: Index | list[str]
    original_columns: Index
    original_dtypes: Series
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
        super().__init__()
        self.groups = groups  # type: ignore[assignment]
        self.dtypes = dtypes  # type: ignore[assignment]
        self.device = device  # type: ignore[assignment]
        self.encode_index = encode_index
        self.inferred_dtypes = {}

    # def __repr__(self) -> str:
    #     return repr_mapping(self.groups, wrapped=self)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[str]:
        return iter(self.groups)

    def __getitem__(self, key: str) -> list[str]:
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
            self.dtypes = {key: self.dtypes for key in self.groups}  # type: ignore[unreachable]
        for key in self.groups:
            val = self.dtypes.get(key, None)
            assert isinstance(val, None | str | torch.dtype)
            self.dtypes[key] = val
            self.inferred_dtypes[key] = (
                TORCH_DTYPES[val] if isinstance(val, str) else val
            )

    def encode(self, data: DataFrame, /) -> dict[str, Tensor]:
        """Encode a DataFrame as a dict of Tensors.

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

    def decode(self, data: dict[str, Tensor | None], /) -> DataFrame:
        dfs = []
        for key, tensor in data.items():
            cols = self.groups[key]
            if tensor is None:
                df = DataFrame(columns=cols).astype(self.original_dtypes[cols])
            else:
                tensor = tensor.clone().detach().cpu()
                df = DataFrame(tensor, columns=cols).astype(self.original_dtypes[cols])
            dfs.append(df)

        # Assemble the DataFrame
        df = pd.concat(dfs, axis="columns")
        df = df.astype(self.original_dtypes[df.columns])
        df = df[self.original_columns.intersection(df.columns)]
        if self.encode_index and self.original_index_columns is not None:
            cols = [col for col in self.original_index_columns if col in df.columns]
            df = df.set_index(cols)
        return df


class FrameAsTuple(BaseEncoder):
    r"""Encodes a DataFrame as a tuple of column and index tensor."""

    requires_fit: ClassVar[bool] = True

    # Attributes
    original_index_columns: Index
    original_index_dtypes: Series
    original_values_columns: Index
    original_values_dtypes: Series

    # Parameters
    column_dtype: Optional[torch.dtype] = None
    index_dtype: Optional[torch.dtype] = None
    device: Optional[str | torch.device] = None

    def __init__(
        self,
        *,
        column_dtype: Optional[torch.dtype] = torch.float32,
        index_dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        self.column_dtype = column_dtype
        self.index_dtype = index_dtype
        self.device = device

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit to the data."""
        index = data.index.to_frame()
        self.original_index_columns = index.columns
        self.original_index_dtypes = index.dtypes
        self.original_values_columns = data.columns
        self.original_values_dtypes = data.dtypes

        if self.original_values_dtypes.nunique() != 1:
            raise ValueError("All columns must have the same dtype!")
        if self.original_index_dtypes.nunique() != 1:
            raise ValueError("All index columns must have the same dtype!")

    def encode(self, data: DataFrame, /) -> tuple[Tensor, Tensor]:
        r"""Encode a DataFrame."""
        index = data.index.to_frame().to_numpy()
        index_tensor = torch.tensor(index, dtype=self.index_dtype, device=self.device)
        values = data.to_numpy()
        values_tensor = torch.tensor(
            values, dtype=self.column_dtype, device=self.device
        )
        return index_tensor.squeeze(), values_tensor.squeeze()

    def decode(self, data: tuple[Tensor, Tensor], /) -> DataFrame:
        r"""Combine index and column tensor to DataFrame."""
        index_tensor, values_tensor = data
        index_tensor = index_tensor.clone().detach().cpu()
        values_tensor = values_tensor.clone().detach().cpu()

        # Assemble the columns
        columns = DataFrame(
            values_tensor,
            columns=self.original_values_columns,
        )
        columns = columns.astype(self.original_values_dtypes)
        columns = columns.squeeze(axis="columns")

        # assemble the index
        index = DataFrame(
            index_tensor,
            columns=self.original_index_columns,
        )
        index = index.astype(self.original_index_dtypes)
        index = index.squeeze(axis="columns")

        if isinstance(index, Series):
            decoded = columns.set_index(index)
        else:
            decoded = columns.set_index(MultiIndex.from_frame(index))
        return decoded


Frame2TensorDict = FrameAsDict  # Do not remove! For old pickle files
"""Alias for FrameAsDict."""
