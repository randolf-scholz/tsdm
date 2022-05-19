r"""Implementation of encoders.

Notes
-----
Contains encoders in modular form.
  - See `tsdm.encoders.functional` for functional implementations.
"""

from __future__ import annotations

__all__ = [
    # ABC
    # Classes
    "ConcatEncoder",
    "DataFrameEncoder",
    "FrameEncoder",
    "FrameIndexer",
    "FrameSplitter",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "PositionalEncoder",
    "SocialTimeEncoder",
    "TensorEncoder",
    "TripletDecoder",
    "TripletEncoder",
    "ValueEncoder",
]

import logging
import warnings
from collections import defaultdict, namedtuple
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Final, Optional, Union, cast, overload

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import NA, DataFrame, DatetimeIndex, Index, MultiIndex, Series
from pandas.core.indexes.frozen import FrozenList
from torch import Tensor

from tsdm.encoders.modular.generic import BaseEncoder
from tsdm.util.strings import repr_mapping
from tsdm.util.torch import TimeTensor
from tsdm.util.types import PandasObject, PathType
from tsdm.util.types.protocols import NTuple

__logger__ = logging.getLogger(__name__)


def apply_along_axes(
    a: Tensor, b: Tensor, op: Callable, axes: tuple[int, ...]
) -> Tensor:
    r"""Apply a function to multiple axes of a tensor.

    Parameters
    ----------
    a: Tensor
    b: Tensor
    op: Callable
    axes: tuple[int, ...]

    Returns
    -------
    Tensor
    """
    axes = tuple(axes)
    rank = len(a.shape)
    source = tuple(range(rank))
    inverse_permutation = axes + tuple(ax for ax in range(rank) if ax not in axes)
    perm = tuple(np.argsort(inverse_permutation))
    if isinstance(a, Tensor):
        a = torch.moveaxis(a, source, perm)
        a = op(a, b)
        a = torch.moveaxis(a, source, inverse_permutation)
    else:
        a = np.moveaxis(a, source, perm)
        a = op(a, b)
        a = np.moveaxis(a, source, inverse_permutation)
    return a


def pairwise_disjoint(sets: Iterable[set]) -> bool:
    r"""Check if sets are pairwise disjoint."""
    union = set().union(*sets)
    return len(union) == sum(len(s) for s in sets)


class CSVEncoder(BaseEncoder):
    r"""Encode the data into a CSV file."""

    filename: PathType
    r"""The filename of the CSV file."""
    dtypes: Series
    r"""The original dtypes."""
    read_csv_kwargs: dict[str, Any]
    r"""The kwargs for the read_csv function."""
    to_csv_kwargs: dict[str, Any]
    r"""The kwargs for the to_csv function."""

    def __init__(
        self,
        filename: PathType,
        to_csv_kwargs: Optional[dict[str, Any]] = None,
        read_csv_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        r"""Initialize the encoder.

        Parameters
        ----------
        filename: str
        to_csv_kwargs: Optional[dict[str, Any]]
        read_csv_kwargs: Optional[dict[str, Any]]
        """
        super().__init__()
        self.filename = filename
        self.read_csv_kwargs = read_csv_kwargs or {}
        self.to_csv_kwargs = to_csv_kwargs or {}

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder.

        Parameters
        ----------
        data
        """
        self.dtypes = data.dtypes

    def encode(self, data: DataFrame, /) -> PathType:
        r"""Encode the data."""
        data.to_csv(self.filename, **self.to_csv_kwargs)
        return self.filename

    def decode(self, data: Optional[PathType] = None, /) -> DataFrame:
        r"""Decode the data."""
        if data is None:
            data = self.filename
        frame = pd.read_csv(data, **self.read_csv_kwargs)
        return DataFrame(frame).astype(self.dtypes)


class ConcatEncoder(BaseEncoder):
    r"""Concatenate multiple encoders."""

    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int

    def __init__(self, axis: int = 0) -> None:
        r"""Concatenate tensors along the specified axis.

        Parameters
        ----------
        axis: int
        """
        super().__init__()
        self.axis = axis

    def fit(self, data: tuple[Tensor, ...], /) -> None:
        r"""Fit to the data."""
        self.numdims = [d.ndim for d in data]
        self.maxdim = max(self.numdims)
        # pad dimensions if necessary
        arrays = [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data]
        # store the lengths of the slices
        self.lengths = [x.shape[self.axis] for x in arrays]

    def encode(self, data: tuple[Tensor, ...], /) -> Tensor:
        r"""Encode the input."""
        return torch.cat(
            [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data], dim=self.axis
        )

    def decode(self, data: Tensor, /) -> tuple[Tensor, ...]:
        r"""Decode the input."""
        result = torch.split(data, self.lengths, dim=self.axis)
        return tuple(x.squeeze() for x in result)


class DataFrameEncoder(BaseEncoder):
    r"""Combine multiple encoders into a single one.

    It is assumed that the DataFrame Modality doesn't change.
    """

    column_encoders: Union[BaseEncoder, Mapping[Any, BaseEncoder]]
    r"""Encoders for the columns."""
    index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None
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
        column_encoders: Union[BaseEncoder, Mapping[Any, BaseEncoder]],
        *,
        index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None,
    ):
        r"""Set up the individual encoders.

        Note: the same encoder instance can be used for multiple columns.

        Parameters
        ----------
        column_encoders
        index_encoders
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

        # add extra repr options by cloning from spec.
        # for x in [
        #     "_repr_data_resource_",
        #     "_repr_fits_horizontal_",
        #     "_repr_fits_vertical_",
        #     "_repr_html_",
        #     "_repr_latex_",
        # ]:
        #     setattr(self, x, getattr(self.spec, x))

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit to the data."""
        self.colspec = data.dtypes

        if self.index_encoders is not None:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            self.index_encoders.fit(data.index)

        if isinstance(self.column_encoders, Mapping):
            # check if cols are a proper partition.
            # keys = set(df.columns)
            # _keys = set(self.column_encoders.keys())
            # assert keys <= _keys, f"Missing encoders for columns {keys - _keys}!"
            # assert (
            #     keys >= _keys
            # ), f"Encoder given for non-existent columns {_keys- keys}!"

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
        r"""Encode the input."""
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
        r"""Decode the input."""
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
        r"""Pretty print."""
        return f"{self.__class__.__name__}(" + self.spec.__repr__() + "\n)"

    def _repr_html_(self) -> str:
        r"""HTML representation."""
        html_repr = self.spec.to_html()
        return f"<h3>{self.__class__.__name__}</h3> {html_repr}"


class FrameEncoder(BaseEncoder):
    r"""Encode a DataFrame by group-wise transformations."""

    columns: Index
    dtypes: Series
    index_columns: Index
    index_dtypes: Series
    duplicate: bool = False

    column_encoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Encoders for the columns."""
    index_encoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Optional Encoder for the index."""
    column_decoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Reverse Dictionary from encoded column name -> encoder"""
    index_decoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Reverse Dictionary from encoded index name -> encoder"""

    @staticmethod
    def _names(
        obj: Union[Index, Series, DataFrame]
    ) -> Union[Hashable, FrozenList[Hashable]]:
        if isinstance(obj, MultiIndex):
            return FrozenList(obj.names)
        if isinstance(obj, (Series, Index)):
            return obj.name
        if isinstance(obj, DataFrame):
            return FrozenList(obj.columns)
        raise ValueError

    def __init__(
        self,
        column_encoders: Optional[
            Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]
        ] = None,
        *,
        index_encoders: Optional[
            Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]
        ] = None,
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
                names = self._names(decoded)
                decoded_cols[names] = decoded

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


# class FrameEncoder(BaseEncoder):
#     r"""Encode a DataFrame by group-wise transformations."""
#
#     columns: Index
#     dtypes: Series
#     index_columns: Index
#     index_dtypes: Series
#     duplicate: bool = False
#
#     column_encoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
#     r"""Encoders for the columns."""
#     index_encoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
#     r"""Optional Encoder for the index."""
#     column_decoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
#     r"""Reverse Dictionary from encoded column name -> encoder"""
#     index_decoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
#     r"""Reverse Dictionary from encoded index name -> encoder"""
#
#     @staticmethod
#     def _names(
#         obj: Union[Index, Series, DataFrame]
#     ) -> Union[Hashable, FrozenList[Hashable]]:
#         if isinstance(obj, MultiIndex):
#             return FrozenList(obj.names)
#         if isinstance(obj, (Series, Index)):
#             return obj.name
#         if isinstance(obj, DataFrame):
#             return FrozenList(obj.columns)
#         raise ValueError
#
#     def __init__(
#         self,
#         column_encoders: Optional[
#             Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]
#         ] = None,
#         *,
#         index_encoders: Optional[
#             Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]
#         ] = None,
#         duplicate: bool = False,
#     ):
#         super().__init__()
#         self.column_encoders = column_encoders
#         self.index_encoders = index_encoders
#         self.duplicate = duplicate
#
#     def fit(self, data: DataFrame, /) -> None:
#         r"""Fit the encoder."""
#         data = data.copy()
#         index = data.index.to_frame()
#         self.columns = data.columns
#         self.dtypes = data.dtypes
#         self.index_columns = index.columns
#         self.index_dtypes = index.dtypes
#
#         if self.duplicate:
#             if not isinstance(self.column_encoders, BaseEncoder):
#                 raise ValueError("Duplication only allowed when single encoder")
#             self.column_encoders = {
#                 col: deepcopy(self.column_encoders) for col in data.columns
#             }
#
#         if self.column_encoders is None:
#             self.column_decoders = None
#         elif isinstance(self.column_encoders, BaseEncoder):
#             self.column_encoders.fit(data)
#             self.column_decoders = self.column_encoders
#         else:
#             self.column_decoders = {}
#             for group, encoder in self.column_encoders.items():
#                 encoder.fit(data[group])
#                 encoded = encoder.encode(data[group])
#                 self.column_decoders[self._names(encoded)] = encoder
#
#         if self.index_encoders is None:
#             self.index_decoders = None
#         elif isinstance(self.index_encoders, BaseEncoder):
#             self.index_encoders.fit(index)
#             self.index_decoders = self.index_encoders
#         else:
#             self.index_decoders = {}
#             for group, encoder in self.index_encoders.items():
#                 encoder.fit(index[group])
#                 encoded = encoder.encode(index[group])
#                 self.index_decoders[self._names(encoded)] = encoder
#
#     def encode(self, data: DataFrame, /) -> DataFrame:
#         r"""Encode the data."""
#         data = data.copy(deep=True)
#         index = data.index.to_frame()
#         encoded_cols = data
#         encoded_inds = encoded_cols.index.to_frame()
#
#         if self.column_encoders is None:
#             pass
#         elif isinstance(self.column_encoders, BaseEncoder):
#             encoded = self.column_encoders.encode(data)
#             encoded_cols = encoded_cols.drop(columns=data.columns)
#             encoded_cols[self._names(encoded)] = encoded
#         else:
#             for group, encoder in self.column_encoders.items():
#                 encoded = encoder.encode(data[group])
#                 encoded_cols = encoded_cols.drop(columns=group)
#                 encoded_cols[self._names(encoded)] = encoded
#
#         if self.index_encoders is None:
#             pass
#         elif isinstance(self.index_encoders, BaseEncoder):
#             encoded = self.index_encoders.encode(index)
#             encoded_inds = encoded_inds.drop(columns=index.columns)
#             encoded_inds[self._names(encoded)] = encoded
#         else:
#             for group, encoder in self.index_encoders.items():
#                 encoded = encoder.encode(index[group])
#                 encoded_inds = encoded_inds.drop(columns=group)
#                 encoded_inds[self._names(encoded)] = encoded
#
#         # Assemble DataFrame
#         encoded = DataFrame(encoded_cols)
#         encoded[self._names(encoded_inds)] = encoded_inds
#         encoded = encoded.set_index(self._names(encoded_inds))
#         return encoded
#
#     def decode(self, data: DataFrame, /) -> DataFrame:
#         r"""Decode the data."""
#         data = data.copy(deep=True)
#         index = data.index.to_frame()
#         decoded_cols = data
#         decoded_inds = decoded_cols.index.to_frame()
#
#         if self.column_decoders is None:
#             pass
#         elif isinstance(self.column_decoders, BaseEncoder):
#             decoded = self.column_decoders.decode(data)
#             decoded_cols = decoded_cols.drop(columns=data.columns)
#             decoded_cols[self._names(decoded)] = decoded
#         else:
#             for group, encoder in self.column_decoders.items():
#                 decoded = encoder.decode(data[group])
#                 decoded_cols = decoded_cols.drop(columns=group)
#                 decoded_cols[self._names(decoded)] = decoded
#
#         if self.index_decoders is None:
#             pass
#         elif isinstance(self.index_decoders, BaseEncoder):
#             decoded = self.index_decoders.decode(index)
#             decoded_inds = decoded_inds.drop(columns=index.columns)
#             decoded_inds[self._names(decoded)] = decoded
#         else:
#             for group, encoder in self.index_decoders.items():
#                 decoded = encoder.decode(index[group])
#                 decoded_inds = decoded_inds.drop(columns=group)
#                 decoded_inds[self._names(decoded)] = decoded
#
#         # Restore index order + dtypes
#         decoded_inds = decoded_inds[self.index_columns]
#         decoded_inds = decoded_inds.astype(self.index_dtypes)
#
#         # Assemble DataFrame
#         decoded = DataFrame(decoded_cols)
#         decoded[self._names(decoded_inds)] = decoded_inds
#         decoded = decoded.set_index(self._names(decoded_inds))
#         decoded = decoded[self.columns]
#         decoded = decoded.astype(self.dtypes)
#
#         return decoded
#
#     def __repr__(self) -> str:
#         r"""Return a string representation of the encoder."""
#         items = {
#             "column_encoders": self.column_encoders,
#             "index_encoders": self.index_encoders,
#         }
#         return repr_mapping(items, title=self.__class__.__name__)


class FrameIndexer(BaseEncoder):
    r"""Change index of a pandas DataFrame.

    For compatibility, this is done by integer index.
    """

    index_columns: Index
    index_dtypes: Series
    index_indices: list[int]
    reset: Union[Hashable, list[Hashable]]

    def __init__(self, *, reset: Optional[Union[Hashable, list[Hashable]]] = None):
        super().__init__()
        if reset is None:
            self.reset = []
        elif reset is Ellipsis:
            self.reset = Ellipsis
        elif isinstance(reset, (str, int, tuple)):
            self.reset = [reset]
        elif isinstance(reset, Iterable):
            self.reset = list(reset)
        else:
            raise TypeError("levels must be None, str, int, tuple or Iterable")

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"{self.__class__.__name__}(levels={self.reset})"

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit to the data."""
        index = data.index.to_frame()
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        # FIXME: EllipsisType py3.10
        if self.reset is Ellipsis or not isinstance(self.reset, list):
            self.index_indices = list(range(len(index.columns)))
        else:
            self.index_indices = list(range(len(self.reset)))

    def encode(self, data: DataFrame, /) -> DataFrame:
        r"""Reset the index."""
        return data.reset_index(level=self.reset)

    def decode(self, data: DataFrame, /) -> DataFrame:
        r"""Set the index."""
        data = DataFrame(data)
        columns = data.columns[self.index_indices].to_list()
        return data.set_index(columns)


class FrameSplitter(BaseEncoder, Mapping):
    r"""Split DataFrame columns into multiple groups.

    The special value `...` (`Ellipsis`) can be used to indicate
    that all other columns belong to this group.

    Index mapping

    [0|1|2|3|4|5]

    [2|0|1], [5|4]

    corresponds to mapping

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

    # FIXME: Union[types.EllipsisType, set[Hashable]] in 3.10
    groups: dict[Hashable, Union[Hashable, list[Hashable]]]
    group_indices: dict[Hashable, list[int]]

    has_ellipsis: bool = False
    ellipsis_columns: Optional[list[Hashable]] = None
    ellipsis: Optional[Hashable] = None

    permutation: list[int]
    inverse_permutation: list[int]
    rtype: type = tuple

    def __init__(
        self,
        groups: Union[Iterable[Hashable], Mapping[Hashable, Hashable]],
        /,
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

    def __repr__(self):
        r"""Return a string representation of the object."""
        return repr_mapping(self)

    def __len__(self):
        r"""Return the number of groups."""
        return len(self.groups)

    def __iter__(self):
        r"""Iterate over the groups."""
        return iter(self.groups)

    def __getitem__(self, item):
        r"""Return the group."""
        return self.groups[item]

    def fit(self, original: DataFrame, /) -> None:
        r"""Fit the encoder."""
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
        self.group_indices: dict[Hashable, list[int]] = {}
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
        r"""Encode the data."""
        data = DataFrame(original).copy()

        if not self.has_ellipsis and set(data.columns) > self.fixed_columns:
            warnings.warn(
                f"Unknown columns {set(data.columns) - self.fixed_columns}."
                "If you want to encode unknown columns add a group `...` (`Ellipsis`)."
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
        r"""Decode the data."""
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


class PositionalEncoder(BaseEncoder):
    r"""Positional encoding.

    .. math::
        x_{2 k}(t)   &:=\sin \left(\frac{t}{t^{2 k / τ}}\right) \\
        x_{2 k+1}(t) &:=\cos \left(\frac{t}{t^{2 k / τ}}\right)

    """

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions."""

    # Buffers
    scale: Final[float]
    r"""Scale factor for positional encoding."""
    scales: Final[np.ndarray]
    r"""Scale factors for positional encoding."""

    def __init__(self, num_dim: int, scale: float) -> None:
        super().__init__()

        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        assert self.scales[0] == 1.0, "Something went wrong."

    def encode(self, data: np.ndarray, /) -> np.ndarray:
        r"""Signature: ``(..., ) -> (..., 2d)``.

        Note: we simple concatenate the sin and cosine terms without interleaving them.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        z = np.einsum("..., d -> ...d", data, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)

    def decode(self, data: np.ndarray, /) -> np.ndarray:
        r"""Signature: ``(..., 2d) -> (..., )``.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        return np.arcsin(data[..., 0])


class TimeSlicer(BaseEncoder):
    r"""Reorganizes the data by slicing."""

    # TODO: multiple horizons

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    @staticmethod
    def is_tensor_pair(data: Any) -> bool:
        r"""Check if the data is a pair of tensors."""
        if isinstance(data, Sequence) and len(data) == 2:
            if isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor):
                return True
        return False

    @overload
    def encode(self, data: TimeTensor, /) -> Sequence[TimeTensor]:
        ...

    @overload
    def encode(self, data: Sequence[TimeTensor], /) -> Sequence[Sequence[TimeTensor]]:
        ...

    @overload
    def encode(
        self,
        data: Sequence[tuple[Tensor, Tensor]],
        /,
    ) -> Sequence[Sequence[tuple[Tensor, Tensor]]]:
        ...

    def encode(self, data, /):
        r"""Slice the data.

        Provide pairs of tensors (T, X) and return a list of pairs (T_sliced, X_sliced).
        """
        if isinstance(data, TimeTensor):
            return data[: self.horizon], data[self.horizon]
        if self.is_tensor_pair(data):
            T, X = data
            idx = T <= self.horizon
            return (T[idx], X[idx]), (T[~idx], X[~idx])
        return tuple(self.encode(item) for item in data)

    @overload
    def decode(self, data: Sequence[TimeTensor], /) -> TimeTensor:
        ...

    @overload
    def decode(self, data: Sequence[Sequence[TimeTensor]], /) -> Sequence[TimeTensor]:
        ...

    @overload
    def decode(
        self,
        data: Sequence[Sequence[tuple[Tensor, Tensor]]],
        /,
    ) -> Sequence[tuple[Tensor, Tensor]]:
        ...

    def decode(self, data, /):
        r"""Restores the original data."""
        if isinstance(data[0], TimeTensor) or self.is_tensor_pair(data[0]):
            return torch.cat(data, dim=0)
        return tuple(self.decode(item) for item in data)


class TripletEncoder(BaseEncoder):
    r"""Encode the data into triplets."""

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
        r"""Initialize the encoder.

        Parameters
        ----------
        sparse: bool = True
        """
        super().__init__()
        self.sparse = sparse
        self.var_name = var_name
        self.value_name = value_name

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder.

        Parameters
        ----------
        data
        """
        self.categories = pd.CategoricalDtype(data.columns)
        self.original_dtypes = data.dtypes
        self.original_columns = data.columns

    def encode(self, data: DataFrame, /) -> DataFrame:
        r"""Encode the data."""
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
        r"""Decode the data."""
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

    categories: pd.CategoricalDtype
    r"""The stored categories."""
    original_dtypes: Series
    r"""The original dtypes."""
    original_columns: Index
    r"""The original columns."""
    value_column: Hashable
    r"""The name of the value column."""
    channel_columns: Union[Index, Hashable]
    r"""The name of the channel column(s)."""

    def __init__(
        self,
        *,
        sparse: bool = False,
        var_name: Optional[str] = None,
        value_name: Optional[str] = None,
    ) -> None:
        r"""Initialize the encoder.

        Parameters
        ----------
        sparse: bool = True
        """
        super().__init__()
        self.sparse = sparse
        self.var_name = var_name
        self.value_name = value_name

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder.

        Parameters
        ----------
        data
        """
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
                f"channel_ids found in '{self.var_name}' does no look like a categoricals!"
                "\n Please specify `value_name` and/or `var_name`!"
            )

        self.categories = pd.CategoricalDtype(np.sort(categories))

    def encode(self, data: DataFrame, /) -> DataFrame:
        r"""Decode the data."""
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
        r"""Encode the data."""
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

    dtype: torch.dtype
    r"""The default dtype."""
    device: torch.device
    r"""The device the tensors are stored in."""
    names: Optional[list[str]] = None
    # colspecs: list[Series]
    # r"""The data types/column names of all the tensors.""
    return_type: type = tuple

    def __init__(
        self,
        names: Optional[list[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.names = names
        self.dtype = torch.float32 if dtype is None else dtype
        # default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") if device is None else device

        if names is not None:
            self.return_type = namedtuple("namedtuple", names)  # type: ignore[misc]

        self.is_fitted = True

    def __repr__(self):
        r"""Pretty print."""
        return f"{self.__class__.__name__}()"

    def fit(self, data: PandasObject, /) -> None:
        r"""Fit to the data."""

    @overload
    def encode(self, data: PandasObject, /) -> Tensor:  # type: ignore[misc]
        ...

    @overload
    def encode(self, data: tuple[PandasObject, ...], /) -> tuple[Tensor, ...]:
        ...

    def encode(self, data, /):
        r"""Convert each inputs to tensor."""
        if isinstance(data, tuple):
            return tuple(self.encode(x) for x in data)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
        if isinstance(data, (Index, Series, DataFrame)):
            return torch.tensor(data.values, device=self.device, dtype=self.dtype)
        return torch.tensor(data, device=self.device, dtype=self.dtype)

    @overload
    def decode(self, data: Tensor, /) -> PandasObject:
        ...

    @overload
    def decode(self, data: tuple[Tensor, ...], /) -> tuple[PandasObject, ...]:
        ...

    def decode(self, data, /):
        r"""Convert each input from tensor to numpy."""
        if isinstance(data, tuple):
            return tuple(self.decode(x) for x in data)
        return data.cpu().numpy()


class ValueEncoder(BaseEncoder):
    r"""Encodes the value of a DataFrame.

    Remembers dtypes, index, columns
    """

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
        r"""Fit to the data."""
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
                UserWarning,
            )

    def encode(self, data: DataFrame, /) -> NDArray:
        r"""Encode the value of a DataFrame."""
        array = data.reset_index().values
        return array.astype(self.dtype)

    def decode(self, data: NDArray, /) -> DataFrame:
        r"""Decode the value of a DataFrame."""
        data = DataFrame(data, columns=self.original_columns)

        # Assemble the columns
        columns = data[self.column_columns]
        columns.columns = self.column_columns
        columns = columns.astype(self.column_dtypes)
        columns = columns.squeeze(axis="columns")

        # assemble the index
        index = data[self.index_columns]
        index.columns = self.index_columns
        index = index.astype(self.index_dtypes)
        index = index.squeeze(axis="columns")

        if isinstance(index, Series):
            decoded = columns.set_index(index)
        else:
            decoded = columns.set_index(MultiIndex.from_frame(index))
        return decoded


# class FrameSplitter(BaseEncoder):
#     r"""Split a DataFrame into multiple groups."""
#
#     columns: Index
#     dtypes: Series
#     groups: dict[Any, Sequence[Any]]
#
#     @staticmethod
#     def _pairwise_disjoint(groups: Iterable[Sequence[HashableType]]) -> bool:
#         union: set[HashableType] = set().union(*(set(obj) for obj in groups))
#         n = sum(len(u) for u in groups)
#         return n == len(union)
#
#     def __init__(self, groups: dict[HashableType, Sequence[HashableType]]) -> None:
#         super().__init__()
#         self.groups = groups
#         assert self._pairwise_disjoint(self.groups.values())
#
#     def fit(self, data: DataFrame, /) -> None:
#         r"""Fit the encoder."""
#         self.columns = data.columns
#         self.dtypes = data.dtypes
#
#     def encode(self, data: DataFrame, /) -> tuple[DataFrame, ...]:
#         r"""Encode the data."""
#         encoded = []
#         for columns in self.groups.values():
#             encoded.append(data[columns].dropna(how="all"))
#         return tuple(encoded)
#
#     def decode(self, data: tuple[DataFrame, ...], /) -> DataFrame:
#         r"""Decode the data."""
#         decoded = pd.concat(data, axis="columns")
#         decoded = decoded.astype(self.dtypes)
#         decoded = decoded[self.columns]
#         return decoded

#
# class FrameSplitter(BaseEncoder, Mapping):
#     r"""Split a DataFrame into multiple groups.
#
#     The special value `...` (`Ellipsis`) can be used to indicate
#     that all other columns belong to this group.
#
#     This function can be used on index columns as well.
#     """
#
#     column_columns: Index
#     column_dtypes: Series
#     column_indices: list[int]
#
#     index_columns: Index
#     index_dtypes = Series
#     index_indices: list[int]
#
#     # FIXME: Union[types.EllipsisType, set[Hashable]] in 3.10
#     groups: dict[Hashable, Union[Hashable, list[Hashable]]]
#     group_indices: dict[Hashable, list[int]]
#
#     indices: dict[Hashable, list[int]]
#     has_ellipsis: bool = False
#     ellipsis: Optional[Hashable] = None
#
#     permutation: list[int]
#     inverse_permutation: list[int]
#
#     # @property
#     # def names(self) -> set[Hashable]:
#     #     r"""Return the union of all groups."""
#     #     sets: list[set] = [
#     #         set(obj) if isinstance(obj, Iterable) else {Ellipsis}
#     #         for obj in self.groups.values()
#     #     ]
#     #     union: set[Hashable] = set.union(*sets)
#     #     assert sum(len(u) for u in sets) == len(union), "Duplicate columns!"
#     #     return union
#
#     def __init__(self, groups: Iterable[Hashable]) -> None:
#         super().__init__()
#
#         if not isinstance(groups, Mapping):
#             groups = dict(enumerate(groups))
#
#         self.groups = {}
#         for key, obj in groups.items():
#             if obj is Ellipsis:
#                 self.groups[key] = obj
#                 self.ellipsis = key
#                 self.has_ellipsis = True
#             elif isinstance(obj, str) or not isinstance(obj, Iterable):
#                 self.groups[key] = [obj]
#             else:
#                 self.groups[key] = list(obj)
#
#     def __repr__(self):
#         r"""Return a string representation of the object."""
#         return repr_mapping(self)
#
#     def __len__(self):
#         r"""Return the number of groups."""
#         return len(self.groups)
#
#     def __iter__(self):
#         r"""Iterate over the groups."""
#         return iter(self.groups)
#
#     def __getitem__(self, item):
#         r"""Return the group."""
#         return self.groups[item]
#
#     def fit(self, data: DataFrame, /) -> None:
#         r"""Fit the encoder."""
#         index = data.index.to_frame()
#         self.column_dtypes = data.dtypes
#         self.column_columns = data.columns
#         self.index_columns = index.columns
#         self.index_dtypes = index.dtypes
#
#         assert not (
#             j := set(self.index_columns) & set(self.column_columns)
#         ), f"index columns and data columns must be disjoint {j}!"
#
#         data = data.copy().reset_index()
#
#         def get_idx(cols: Any) -> list[int]:
#             return [data.columns.get_loc(i) for i in cols]
#
#         self.indices: dict[Hashable, int] = dict(enumerate(data.columns))
#         self.group_indices: dict[Hashable, list[int]] = {}
#         self.column_indices = get_idx(self.column_columns)
#         self.index_indices = get_idx(self.index_columns)
#
#         # replace ellipsis indices
#         if self.has_ellipsis:
#             # FIXME EllipsisType in 3.10
#             fixed_cols = set().union(
#                 *(
#                     set(cols)  # type: ignore[arg-type]
#                     for cols in self.groups.values()
#                     if cols is not Ellipsis
#                 )
#             )
#             ellipsis_columns = [c for c in data.columns if c not in fixed_cols]
#             self.groups[self.ellipsis] = ellipsis_columns
#
#         # set column indices
#         self.permutation = []
#         for group, columns in self.groups.items():
#             if columns is Ellipsis:
#                 continue
#             self.group_indices[group] = get_idx(columns)
#             self.permutation += self.group_indices[group]
#         self.inverse_permutation = np.argsort(self.permutation).tolist()
#         # sorted(p.copy(), key=p.__getitem__)
#
#     def encode(self, data: DataFrame, /) -> tuple[DataFrame, ...]:
#         r"""Encode the data."""
#         # copy the frame and add index as columns.
#         data = data.reset_index()  # prepend index as columns!
#         data_columns = set(data.columns)
#
#         assert data_columns <= set(self.indices.values()), (
#             f"Unknown columns {data_columns - set(self.indices)}."
#             "If you want to encode unknown columns add a group `...` (Ellipsis)."
#         )
#
#         encoded = []
#         for columns in self.groups.values():
#             encoded.append(data[columns].squeeze(axis="columns"))
#         return tuple(encoded)
#
#     def decode(self, data: tuple[DataFrame, ...], /) -> DataFrame:
#         r"""Decode the data."""
#         data = tuple(DataFrame(x) for x in data)
#         joined = pd.concat(data, axis="columns")
#
#         # unshuffle the columns, restoring original order
#         joined = joined.iloc[..., self.inverse_permutation]
#
#         # Assemble the columns
#         columns = joined.iloc[..., self.column_indices]
#         columns.columns = self.column_columns
#         columns = columns.astype(self.column_dtypes)
#         columns = columns.squeeze(axis="columns")
#
#         # assemble the index
#         index = joined.iloc[..., self.index_indices]
#         index.columns = self.index_columns
#         index = index.astype(self.index_dtypes)
#         index = index.squeeze(axis="columns")
#
#         if isinstance(index, Series):
#             decoded = columns.set_index(index)
#         else:
#             decoded = columns.set_index(MultiIndex.from_frame(index))
#         return decoded


class PeriodicEncoder(BaseEncoder):
    r"""Encode periodic data as sin/cos waves."""

    period: float
    freq: float
    dtype: pd.dtype
    colname: Hashable

    def __init__(self, period: Optional[float] = None) -> None:
        super().__init__()
        self._period = period

    def fit(self, x: Series) -> None:
        r"""Fit the encoder."""
        self.dtype = x.dtype
        self.colname = x.name
        self.period = x.max() + 1 if self._period is None else self._period
        self._period = self.period
        self.freq = 2 * np.pi / self.period

    def encode(self, x: Series) -> DataFrame:
        r"""Encode the data."""
        x = self.freq * (x % self.period)  # ensure 0...N-1
        return DataFrame(
            np.stack([np.cos(x), np.sin(x)]).T,
            columns=[f"cos_{self.colname}", f"sin_{self.colname}"],
        )

    def decode(self, x: DataFrame) -> Series:
        r"""Decode the data."""
        x = np.arctan2(x[f"sin_{self.colname}"], x[f"cos_{self.colname}"])
        x = (x / self.freq) % self.period
        return Series(x, dtype=self.dtype, name=self.colname)

    def __repr__(self):
        """Pretty-print."""
        return f"{self.__class__.__name__}({self._period})"


class SocialTimeEncoder(BaseEncoder):
    r"""Social time encoding."""

    level_codes = {
        "Y": "year",
        "M": "month",
        "W": "weekday",
        "D": "day",
        "h": "hour",
        "m": "minute",
        "s": "second",
        "µ": "microsecond",
        "n": "nanosecond",
    }
    original_name: Hashable
    original_dtype: pd.dtype
    original_type: type
    rev_cols: list[str]
    level_code: str
    levels: list[str]

    def __init__(self, levels: str = "YMWDhms") -> None:
        super().__init__()
        self.level_code = levels
        self.levels = [self.level_codes[k] for k in levels]

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.original_type = type(x)
        self.original_name = x.name
        self.original_dtype = x.dtype
        self.rev_cols = [level for level in self.levels if level != "weekday"]

    def encode(self, x: Series, /) -> DataFrame:
        r"""Encode the data."""
        if isinstance(x, DatetimeIndex):
            res = {level: getattr(x, level) for level in self.levels}
        else:
            res = {level: getattr(x, level) for level in self.levels}
        return DataFrame.from_dict(res)

    def decode(self, x: DataFrame, /) -> Series:
        r"""Decode the data."""
        x = x[self.rev_cols]
        s = pd.to_datetime(x)
        return self.original_type(s, name=self.original_name, dtype=self.original_dtype)

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"SocialTimeEncoder('{self.level_code}')"


class PeriodicSocialTimeEncoder(SocialTimeEncoder):
    r"""Combines SocialTimeEncoder with PeriodicEncoder using the right frequencies."""

    frequencies = {
        "year": 1,
        "month": 12,
        "weekday": 7,
        "day": 365,
        "hour": 24,
        "minute": 60,
        "second": 60,
        "microsecond": 1000,
        "nanosecond": 1000,
    }
    """The frequencies of the used `PeriodicEncoder`."""

    def __new__(cls, levels: str = "YMWDhms") -> PeriodicSocialTimeEncoder:
        r"""Construct a new encoder object."""
        self = super().__new__(cls)
        self.__init__(levels)  # type: ignore[misc]
        column_encoders = {
            level: PeriodicEncoder(period=self.frequencies[level])
            for level in self.levels
        }
        obj = FrameEncoder(column_encoders) @ self  # type: ignore[arg-type]
        return cast(PeriodicSocialTimeEncoder, obj)
