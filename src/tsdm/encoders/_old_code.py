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
