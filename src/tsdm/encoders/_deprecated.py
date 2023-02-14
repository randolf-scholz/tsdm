"""Deprecated Encoders."""

__all__ = [
    # Classes
    "DataFrameEncoder",
]


from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Sequence
from copy import deepcopy
from types import EllipsisType
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from pandas import NA, DataFrame, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList
from torch import Tensor

from tsdm.encoders.base import BaseEncoder
from tsdm.types.abc import HashableType
from tsdm.types.variables import AnyVar as T
from tsdm.utils.strings import repr_mapping


class DataFrameEncoder(BaseEncoder):
    r"""Combine multiple encoders into a single one.

    It is assumed that the DataFrame Modality doesn't change.
    """

    column_encoders: BaseEncoder | Mapping[Any, BaseEncoder]
    r"""Encoders for the columns."""
    index_encoders: Optional[BaseEncoder | Mapping[Any, BaseEncoder]] = None
    r"""Optional Encoder for the index."""
    colspec: Series
    r"""The columns-specification of the DataFrame."""
    encode_index: bool
    r"""Whether to encode the index."""
    column_wise: bool
    r"""Whether to encode column-wise."""
    partitions: Optional[dict] = None
    r"""Contains partitions if used column wise."""

    def __init__(
        self,
        column_encoders: BaseEncoder | Mapping[Any, BaseEncoder],
        *,
        index_encoders: Optional[BaseEncoder | Mapping[Any, BaseEncoder]] = None,
    ):
        r"""Set up the individual encoders.

        Note: the same encoder instance can be used for multiple columns.
        """
        super().__init__()
        self.column_encoders = column_encoders

        if isinstance(index_encoders, Mapping):
            raise NotImplementedError("Multi-Index encoders not yet supported")

        self.index_encoders = index_encoders
        self.column_wise: bool = isinstance(self.column_encoders, Mapping)
        self.encode_index: bool = index_encoders is not None

        index_spec = DataFrame(
            columns=["col", "encoder"],
            index=Index([], name="partition"),
        )

        if self.encode_index:
            if not isinstance(self.index_encoders, Mapping):
                _idxenc_spec = Series(
                    {
                        "col": NA,
                        "encoder": self.index_encoders,
                    },
                    name=0,
                )
                # index_spec = index_spec.append(_idxenc_spec)
                index_spec.loc[0] = _idxenc_spec
            else:
                raise NotImplementedError(
                    "Multiple Index encoders are not supported yet."
                )

        if not isinstance(self.column_encoders, Mapping):
            colenc_spec = DataFrame(
                columns=["col", "encoder"],
                index=Index([], name="partition"),
            )

            _colenc_spec = Series(
                {
                    "col": NA,
                    "encoder": self.column_encoders,
                },
                name=0,
            )
            # colenc_spec = colenc_spec.append(_colenc_spec)
            # colenc_spec = pandas.concat([colenc_spec, _colenc_spec])
            colenc_spec.loc[0] = _colenc_spec
        else:
            keys = self.column_encoders.keys()
            assert len(set(keys)) == len(keys), "Some index are duplicates!"

            encoders = Series(self.column_encoders.values(), name="encoder")
            partitions = Series(range(len(encoders)), name="partition")

            _columns: dict = defaultdict(list)
            for key, encoder in self.column_encoders.items():
                if isinstance(key, str):
                    _columns[encoder] = key
                else:
                    _columns[encoder].extend(key)

            columns = Series(_columns, name="col")
            colenc_spec = DataFrame(encoders, index=partitions)
            colenc_spec = colenc_spec.join(columns, on="encoder")

        self.spec = pd.concat(
            [index_spec, colenc_spec],
            keys=["index", "columns"],
            names=["section", "partition"],
        ).astype({"col": object})

        self.spec.name = self.__class__.__name__

    def fit(self, data: DataFrame, /) -> None:
        self.colspec = data.dtypes

        if self.index_encoders is not None:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            self.index_encoders.fit(data.index)

        if isinstance(self.column_encoders, Mapping):
            for _, series in self.spec.loc["columns"].iterrows():
                encoder = series["encoder"]
                cols = series["col"]
                encoder.fit(data[cols])
        else:
            cols = list(data.columns)
            self.spec.loc["columns"].iloc[0]["col"] = cols
            encoder = self.spec.loc["columns", "encoder"].item()
            encoder.fit(data)

    def encode(self, data: DataFrame, /) -> tuple:
        tensors = []
        for _, series in self.spec.loc["columns"].iterrows():
            encoder = series["encoder"]
            cols = series["col"]
            tensors.append(encoder.encode(data[cols]))
        encoded_columns = tuple(tensors)

        if self.index_encoders is not None:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            encoded_index = self.index_encoders.encode(data.index)
            return encoded_index, *encoded_columns
        return encoded_columns

    def decode(self, data: tuple, /) -> DataFrame:
        if self.encode_index:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            encoder = self.spec.loc["index", "encoder"].item()
            index = encoder.decode(data[0])
            data = data[1:]
        else:
            index = None

        columns = []
        col_names = []
        for partition, (col_name, encoder) in self.spec.loc["columns"].iterrows():
            tensor = data[partition]
            columns.append(encoder.decode(tensor))
            if isinstance(col_name, str):
                col_names.append(col_name)
            else:
                col_names.extend(col_name)

        columns = [
            np.expand_dims(arr, axis=1) if arr.ndim < 2 else arr for arr in columns
        ]
        values = np.concatenate(columns, axis=1)
        df = DataFrame(values, index=index, columns=col_names)
        return df[self.colspec.index].astype(self.colspec)  # bring cols in right order

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + self.spec.__repr__() + "\n)"

    def _repr_html_(self) -> str:
        r"""HTML representation."""
        html_repr = self.spec.to_html()
        return f"<h3>{self.__class__.__name__}</h3> {html_repr}"


class _DeprecatedFrame2Tensor(BaseEncoder):
    r"""Encodes a DataFrame as a tuple of column and index tensor."""

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
        index = data.index.to_frame().to_numpy()
        index_tensor = torch.tensor(index, dtype=self.index_dtype, device=self.device)
        values = data.to_numpy()
        values_tensor = torch.tensor(
            values, dtype=self.column_dtype, device=self.device
        )
        return index_tensor.squeeze(), values_tensor.squeeze()

    def decode(self, data: tuple[Tensor, Tensor], /) -> DataFrame:
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


class FrameSplitter(BaseEncoder):
    r"""Split a DataFrame into multiple groups."""

    columns: Index
    dtypes: Series
    groups: dict[Any, Sequence[Any]]

    @staticmethod
    def _pairwise_disjoint(groups: Iterable[Sequence[HashableType]]) -> bool:
        union: set[HashableType] = set().union(*(set(obj) for obj in groups))
        n = sum(len(u) for u in groups)
        return n == len(union)

    def __init__(self, groups: dict[T, Sequence[HashableType]]) -> None:
        super().__init__()
        self.groups = groups
        assert self._pairwise_disjoint(self.groups.values())

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder."""
        self.columns = data.columns
        self.dtypes = data.dtypes

    def encode(self, data: DataFrame, /) -> tuple[DataFrame, ...]:
        r"""Encode the data."""
        encoded = []
        for columns in self.groups.values():
            encoded.append(data[columns].dropna(how="all"))
        return tuple(encoded)

    def decode(self, data: tuple[DataFrame, ...], /) -> DataFrame:
        r"""Decode the data."""
        decoded = pd.concat(data, axis="columns")
        decoded = decoded.astype(self.dtypes)
        decoded = decoded[self.columns]
        return decoded


class FrameSplitter2(BaseEncoder, Mapping):
    r"""Split a DataFrame into multiple groups.

    The special value `...` (`Ellipsis`) can be used to indicate
    that all other columns belong to this group.

    This function can be used on index columns as well.
    """

    column_columns: Index
    column_dtypes: Series
    column_indices: list[int]

    index_columns: Index
    index_dtypes = Series
    index_indices: list[int]

    groups: dict[Any, EllipsisType | Hashable | list[Hashable]]
    group_indices: dict[Any, list[int]]

    indices: dict[Any, list[int]]
    has_ellipsis: bool = False
    ellipsis: Optional[Hashable] = None

    permutation: list[int]
    inverse_permutation: list[int]

    def __init__(self, groups: Iterable[Hashable]) -> None:
        super().__init__()

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

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return repr_mapping(self)

    def __len__(self) -> int:
        r"""Return the number of groups."""
        return len(self.groups)

    def __iter__(self) -> Iterator[Any]:
        r"""Iterate over the groups."""
        return iter(self.groups)

    def __getitem__(self, item: Any) -> Any:
        r"""Return the group."""
        return self.groups[item]

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder."""
        index = data.index.to_frame()
        self.column_dtypes = data.dtypes
        self.column_columns = data.columns
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        assert not (
            j := set(self.index_columns) & set(self.column_columns)
        ), f"index columns and data columns must be disjoint {j}!"

        data = data.copy().reset_index()

        def get_idx(cols: Any) -> list[int]:
            return [data.columns.get_loc(i) for i in cols]

        self.indices: dict[Any, int] = dict(enumerate(data.columns))
        self.group_indices: dict[Any, list[int]] = {}
        self.column_indices = get_idx(self.column_columns)
        self.index_indices = get_idx(self.index_columns)

        # replace ellipsis indices
        if self.has_ellipsis:
            fixed_cols = set().union(
                *(
                    set(cols)  # type: ignore[arg-type]
                    for cols in self.groups.values()
                    if cols is not Ellipsis
                )
            )
            ellipsis_columns = [c for c in data.columns if c not in fixed_cols]
            self.groups[self.ellipsis] = ellipsis_columns

        # set column indices
        self.permutation = []
        for group, columns in self.groups.items():
            if columns is Ellipsis:
                continue
            self.group_indices[group] = get_idx(columns)
            self.permutation += self.group_indices[group]
        self.inverse_permutation = np.argsort(self.permutation).tolist()
        # sorted(p.copy(), key=p.__getitem__)

    def encode(self, data: DataFrame, /) -> tuple[DataFrame, ...]:
        r"""Encode the data."""
        # copy the frame and add index as columns.
        data = data.reset_index()  # prepend index as columns!
        data_columns = set(data.columns)

        assert data_columns <= set(self.indices.values()), (
            f"Unknown columns {data_columns - set(self.indices)}."
            "If you want to encode unknown columns add a group `...` (Ellipsis)."
        )

        encoded = []
        for columns in self.groups.values():
            encoded.append(data[columns].squeeze(axis="columns"))
        return tuple(encoded)

    def decode(self, data: tuple[DataFrame, ...], /) -> DataFrame:
        r"""Decode the data."""
        data = tuple(DataFrame(x) for x in data)
        joined = pd.concat(data, axis="columns")

        # unshuffle the columns, restoring original order
        joined = joined.iloc[..., self.inverse_permutation]

        # Assemble the columns
        columns = joined.iloc[..., self.column_indices]
        columns.columns = self.column_columns
        columns = columns.astype(self.column_dtypes)
        columns = columns.squeeze(axis="columns")

        # assemble the index
        index = joined.iloc[..., self.index_indices]
        index.columns = self.index_columns
        index = index.astype(self.index_dtypes)
        index = index.squeeze(axis="columns")

        if isinstance(index, Series):
            decoded = columns.set_index(index)
        else:
            decoded = columns.set_index(MultiIndex.from_frame(index))
        return decoded


class FrameEncoder(BaseEncoder):
    r"""Encode a DataFrame by group-wise transformations."""

    columns: Index
    dtypes: Series
    index_columns: Index
    index_dtypes: Series
    duplicate: bool = False

    column_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Encoders for the columns."""
    index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Optional Encoder for the index."""
    column_decoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Reverse Dictionary from encoded column name -> encoder"""
    index_decoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Reverse Dictionary from encoded index name -> encoder"""

    @staticmethod
    def _names(obj: Index | Series | DataFrame) -> Hashable | FrozenList[Hashable]:
        if isinstance(obj, MultiIndex):
            return FrozenList(obj.names)
        if isinstance(obj, Series | Index):
            return obj.name
        if isinstance(obj, DataFrame):
            return FrozenList(obj.columns)
        raise ValueError

    def __init__(
        self,
        column_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None,
        *,
        index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None,
        duplicate: bool = False,
    ):
        super().__init__()
        self.column_encoders = column_encoders
        self.index_encoders = index_encoders
        self.duplicate = duplicate

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder."""
        data = data.copy()
        index = data.index.to_frame()
        self.columns = data.columns
        self.dtypes = data.dtypes
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        if self.duplicate:
            if not isinstance(self.column_encoders, BaseEncoder):
                raise ValueError("Duplication only allowed when single encoder")
            self.column_encoders = {
                col: deepcopy(self.column_encoders) for col in data.columns
            }

        if self.column_encoders is None:
            self.column_decoders = None
        elif isinstance(self.column_encoders, BaseEncoder):
            self.column_encoders.fit(data)
            self.column_decoders = self.column_encoders
        else:
            self.column_decoders = {}
            for group, encoder in self.column_encoders.items():
                encoder.fit(data[group])
                encoded = encoder.encode(data[group])
                self.column_decoders[self._names(encoded)] = encoder

        if self.index_encoders is None:
            self.index_decoders = None
        elif isinstance(self.index_encoders, BaseEncoder):
            self.index_encoders.fit(index)
            self.index_decoders = self.index_encoders
        else:
            self.index_decoders = {}
            for group, encoder in self.index_encoders.items():
                encoder.fit(index[group])
                encoded = encoder.encode(index[group])
                self.index_decoders[self._names(encoded)] = encoder

    def encode(self, data: DataFrame, /) -> DataFrame:
        r"""Encode the data."""
        data = data.copy(deep=True)
        index = data.index.to_frame()
        encoded_cols = data
        encoded_inds = encoded_cols.index.to_frame()

        if self.column_encoders is None:
            pass
        elif isinstance(self.column_encoders, BaseEncoder):
            encoded = self.column_encoders.encode(data)
            encoded_cols = encoded_cols.drop(columns=data.columns)
            encoded_cols[self._names(encoded)] = encoded
        else:
            for group, encoder in self.column_encoders.items():
                encoded = encoder.encode(data[group])
                encoded_cols = encoded_cols.drop(columns=group)
                encoded_cols[self._names(encoded)] = encoded

        if self.index_encoders is None:
            pass
        elif isinstance(self.index_encoders, BaseEncoder):
            encoded = self.index_encoders.encode(index)
            encoded_inds = encoded_inds.drop(columns=index.columns)
            encoded_inds[self._names(encoded)] = encoded
        else:
            for group, encoder in self.index_encoders.items():
                encoded = encoder.encode(index[group])
                encoded_inds = encoded_inds.drop(columns=group)
                encoded_inds[self._names(encoded)] = encoded

        # Assemble DataFrame
        encoded = DataFrame(encoded_cols)
        encoded[self._names(encoded_inds)] = encoded_inds
        encoded = encoded.set_index(self._names(encoded_inds))
        return encoded

    def decode(self, data: DataFrame, /) -> DataFrame:
        r"""Decode the data."""
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
        else:
            for group, encoder in self.column_decoders.items():
                decoded = encoder.decode(data[group])
                decoded_cols = decoded_cols.drop(columns=group)
                decoded_cols[self._names(decoded)] = decoded

        if self.index_decoders is None:
            pass
        elif isinstance(self.index_decoders, BaseEncoder):
            decoded = self.index_decoders.decode(index)
            decoded_inds = decoded_inds.drop(columns=index.columns)
            decoded_inds[self._names(decoded)] = decoded
        else:
            for group, encoder in self.index_decoders.items():
                decoded = encoder.decode(index[group])
                decoded_inds = decoded_inds.drop(columns=group)
                decoded_inds[self._names(decoded)] = decoded

        # Restore index order + dtypes
        decoded_inds = decoded_inds[self.index_columns]
        decoded_inds = decoded_inds.astype(self.index_dtypes)

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
        return repr_mapping(items, title=self.__class__.__name__)


class FastFrameEncoder(BaseEncoder):
    r"""Encode a DataFrame by group-wise transformations.

    This variant ensures that the output and input DataFrame have the same modality.
    """

    columns: Index
    dtypes: Series
    index_columns: Index
    index_dtypes: Series
    duplicate: bool = False

    column_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Encoders for the columns."""
    index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Optional Encoder for the index."""
    column_decoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Reverse Dictionary from encoded column name -> encoder"""
    index_decoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]]
    r"""Reverse Dictionary from encoded index name -> encoder"""

    @staticmethod
    def _names(obj: Index | Series | DataFrame) -> Hashable | FrozenList[Hashable]:
        if isinstance(obj, MultiIndex):
            return FrozenList(obj.names)
        if isinstance(obj, Series | Index):
            return obj.name
        if isinstance(obj, DataFrame):
            return FrozenList(obj.columns)
        raise ValueError

    def __init__(
        self,
        column_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None,
        *,
        index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None,
        duplicate: bool = False,
    ):
        super().__init__()
        self.column_encoders = column_encoders
        self.index_encoders = index_encoders
        self.duplicate = duplicate

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder."""
        data = data.copy()
        index = data.index.to_frame()
        self.columns = data.columns
        self.dtypes = data.dtypes
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        if self.duplicate:
            if not isinstance(self.column_encoders, BaseEncoder):
                raise ValueError("Duplication only allowed when single encoder")
            self.column_encoders = {
                col: deepcopy(self.column_encoders) for col in data.columns
            }

        if self.column_encoders is None:
            self.column_decoders = None
        elif isinstance(self.column_encoders, BaseEncoder):
            self.column_encoders.fit(data)
            self.column_decoders = self.column_encoders
        else:
            self.column_decoders = {}
            for group, encoder in self.column_encoders.items():
                encoder.fit(data[group])
                encoded = encoder.encode(data[group])
                self.column_decoders[self._names(encoded)] = encoder

        if self.index_encoders is None:
            self.index_decoders = None
        elif isinstance(self.index_encoders, BaseEncoder):
            self.index_encoders.fit(index)
            self.index_decoders = self.index_encoders
        else:
            self.index_decoders = {}
            for group, encoder in self.index_encoders.items():
                encoder.fit(index[group])
                encoded = encoder.encode(index[group])
                self.index_decoders[self._names(encoded)] = encoder

    def encode(self, data: DataFrame, /) -> DataFrame:
        r"""Encode the data."""
        data = data.copy()
        index = data.index.to_frame()

        if self.column_encoders is None:
            pass
        elif isinstance(self.column_encoders, BaseEncoder):
            data[:] = self.column_encoders.encode(data)
        else:
            for group, encoder in self.column_encoders.items():
                data[group] = encoder.encode(data[group])

        if self.index_encoders is None:
            pass
        elif isinstance(self.index_encoders, BaseEncoder):
            index[:] = self.index_encoders.encode(index)
        else:
            for group, encoder in self.index_encoders.items():
                index[group] = encoder.encode(index[group])

        # Assemble DataFrame
        data.index = index
        return data

    def decode(self, data: DataFrame, /) -> DataFrame:
        r"""Decode the data."""
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
        else:
            for group, encoder in self.column_decoders.items():
                decoded = encoder.decode(data[group])
                decoded_cols = decoded_cols.drop(columns=group)
                decoded_cols[self._names(decoded)] = decoded

        if self.index_decoders is None:
            pass
        elif isinstance(self.index_decoders, BaseEncoder):
            decoded = self.index_decoders.decode(index)
            decoded_inds = decoded_inds.drop(columns=index.columns)
            decoded_inds[self._names(decoded)] = decoded
        else:
            for group, encoder in self.index_decoders.items():
                decoded = encoder.decode(index[group])
                decoded_inds = decoded_inds.drop(columns=group)
                decoded_inds[self._names(decoded)] = decoded

        # Restore index order + dtypes
        decoded_inds = decoded_inds[self.index_columns]
        decoded_inds = decoded_inds.astype(self.index_dtypes)

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
        return repr_mapping(items, title=self.__class__.__name__)
