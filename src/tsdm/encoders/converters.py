r"""Encoders for converting between backends."""

__all__ = [
    "NDArrayToTensor",
    "FrameAsTensor",
    "FrameAsTensorDict",
    "FrameDTypeConverter",
    "FrameAsDict",
    "get_ellipsis_cols",
]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from numpy._typing import NDArray
from pandas import DataFrame
from torch import Tensor

from tsdm.constants import UNDEFINED
from tsdm.pprint import pprint_repr
from tsdm.types.nested import NestedBuiltin
from tsdm.utils.funcutils import recurse_on_nested_builtin

from .base import FittableEncoder, StaticEncoder

type PandasDType = Any
type PandasDTypeArg = str | type | PandasDType


class NDArrayToTensor(StaticEncoder[NestedBuiltin[NDArray], NestedBuiltin[Tensor]]):
    r"""Encodes nested data as tensors."""

    def encode(self, x: NestedBuiltin[NDArray], /) -> NestedBuiltin[Tensor]:
        return recurse_on_nested_builtin(x, leaf_fn=torch.tensor, leaf_type=np.ndarray)

    def decode(self, y: NestedBuiltin[Tensor], /) -> NestedBuiltin[NDArray]:
        return recurse_on_nested_builtin(y, leaf_fn=Tensor.numpy, leaf_type=Tensor)  # pyright: ignore[reportArgumentType]


@pprint_repr
@dataclass(slots=True)
class FrameAsTensor(FittableEncoder[DataFrame, Tensor]):
    r"""Converts a `DataFrame` to a `torch.Tensor`.

    Note:
        - This encoder requires that the DataFrame is canonically indexed. (i.e. `index = range(len(df))`)
        - This encoder requires that the DataFrame is of a single (numerical) dtype.
    """

    dtype: Optional[torch.dtype] = None
    r"""The default dtype."""
    device: Optional[str | torch.device] = None
    r"""The device the tensors are stored in."""
    original_schema: Mapping[str, Any] = UNDEFINED
    r"""The original schema."""

    def fit(self, data: DataFrame, /) -> None:
        if data.index != pd.RangeIndex(len(data)):
            raise ValueError("DataFrame must be canonically indexed!")
        self.original_schema = data.dtypes.to_dict()

    def encode(self, data: DataFrame, /) -> Tensor:
        return torch.tensor(data.values, device=self.device, dtype=self.dtype)

    def decode(self, data: Tensor, /) -> DataFrame:
        array = data.detach().cpu().numpy()
        return DataFrame(array, columns=self.original_schema).astype(
            self.original_schema
        )


@pprint_repr
@dataclass(slots=True, init=False)
class FrameAsTensorDict(FittableEncoder[DataFrame, dict[str, Tensor]]):
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
        >>> df = DataFrame(
        ...     {
        ...         "ID": [10, 21, 33],
        ...         "mask": [True, False, False],
        ...         "x": [-2.1, 7.3, 3.5],
        ...         "y": [0.1, 0.2, 0.3],
        ...     }
        ... ).set_index("ID")
        >>> encoder = FrameAsTensorDict(
        ...     {
        ...         "index": "ID",
        ...         "mask": "mask",
        ...         "features": ["x", "y"],
        ...     }
        ... )
        >>> encoder.fit(df)
        >>> encoded = encoder.encode(df)
        >>> assert isinstance(encoded, dict)
        >>> decoded = encoder.decode(encoded)
        >>> pd.testing.assert_frame_equal(df, decoded)
    """

    # Attributes (type hints represent post-fit attributes)
    schema: Mapping[str, list[str] | EllipsisType]
    r"""The schema for grouping the columns (group-name -> col-name(s))."""
    device: dict[str | EllipsisType, None | str | torch.device]
    r"""The device for each group (group-name -> device)."""
    dtypes: dict[str | EllipsisType, None | torch.dtype]
    r"""The dtype for each group (group-name -> dtype)."""

    # Fitted attributes
    original_index: list[str] = UNDEFINED
    original_schema: Mapping[str, Any] = UNDEFINED  # cols -> dtype
    target_schema: Mapping[str, list[str]] = UNDEFINED

    def __init__(
        self,
        schema: Mapping[str, str | list[str] | EllipsisType],
        *,
        device: Optional[
            str | torch.device | Mapping[str | EllipsisType, None | str | torch.device]
        ] = None,
        dtypes: Optional[
            torch.dtype | Mapping[str | EllipsisType, None | torch.dtype]
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
        self.original_index = list(data.index.names)

        # reset the index if it is not the default index
        match data.index:
            case pd.RangeIndex(start=0, step=1, stop=stop) if stop == len(data):
                pass
            case _:
                data = data.reset_index()

        # get the original dtypes
        self.original_schema = data.dtypes.to_dict()
        self.target_schema = {}
        ellipsis_cols: list[str] = get_ellipsis_cols(data, self.schema.values())
        for group, cols in self.schema.items():
            if cols is ...:  # NOTE: https://github.com/microsoft/pyright/issues/10721
                self.target_schema[group] = ellipsis_cols
            else:
                self.target_schema[group] = cols
        assert self.target_schema.keys() == self.schema.keys()

        # fill in the dtype for missing groups
        dtype = None if Ellipsis not in self.dtypes else self.dtypes.pop(Ellipsis)
        for group in self.target_schema.keys() - self.dtypes.keys():
            self.dtypes[group] = dtype

        # fill in the device for missing groups
        device = None if Ellipsis not in self.device else self.device.pop(Ellipsis)
        for group in self.target_schema.keys() - self.device.keys():
            self.device[group] = device

        if self.dtypes.keys() & self.device.keys() != self.target_schema.keys():
            raise ValueError(
                "Schema, dtypes and device columns must share groups!"
                f"\nSchema: {self.target_schema}"
                f"\ndtypes: {self.dtypes}"
                f"\ndevice: {self.device}"
            )

        if missing_cols := set().union(*self.target_schema.values()) - set(
            data.columns
        ):
            raise ValueError(f"Missing columns {missing_cols}!")
        if extra_cols := set(data.columns) - set().union(*self.target_schema.values()):
            raise ValueError(f"Extra columns {extra_cols}!")

    def encode(self, data: DataFrame, /) -> dict[str, Tensor]:
        r"""Encode a DataFrame as a dict of Tensors.

        The encode method ensures treatment of missingness:
        if columns in the dataframe are missing, the correponding tensor columns
        will be filled with `NAN`-values if the datatype allows it.
        """
        data = data.reset_index()
        return {
            # FIXME: https://github.com/pandas-dev/pandas/issues/22791
            group: torch.tensor(
                np.stack([data[col].to_numpy() for col in cols], axis=-1),
                device=self.device[group],
                dtype=self.dtypes[group],
            ).squeeze()
            for group, cols in self.target_schema.items()
        }

    def decode(self, data: Mapping[str, Tensor], /) -> DataFrame:
        # convert the tensors to dataframes
        dfs = [
            DataFrame(tensor.detach().cpu().numpy(), columns=self.target_schema[group])
            for group, tensor in data.items()
        ]

        # Assemble the DataFrame
        df = (
            pd.concat(dfs, axis="columns")
            # restores column order / adds missing columns
            .reindex(columns=self.original_schema)
            # restore original dtypes
            .astype(self.original_schema)
        )

        if self.original_index != [None]:
            df = df.set_index(self.original_index)

        return df


@pprint_repr
@dataclass(slots=True, init=False)
class FrameDTypeConverter(FittableEncoder[DataFrame, DataFrame]):
    r"""Converts dtypes of a DataFrame.

    Note:
        - If a column is not present, it will be ignored.
        - If `...` (`Ellipsis`) is given,
          all unspecified columns will be converted to the given dtype.
          (Ellipsis will be removed during `.fit()`)
    """

    target_dtypes: dict[Any, PandasDTypeArg] = UNDEFINED
    r"""The target dtypes."""
    original_schema: dict[str, Any] = UNDEFINED
    r"""The original dtypes."""

    def __init__(self, dtypes: PandasDTypeArg | Mapping[Any, PandasDTypeArg]) -> None:
        super().__init__()
        self.target_dtypes = (
            dict(dtypes) if isinstance(dtypes, Mapping) else {...: dtypes}
        )

    def fit(self, data: DataFrame, /) -> None:
        self.original_schema = data.dtypes.to_dict()

        if Ellipsis in self.target_dtypes:
            fill_dtype = self.target_dtypes.pop(Ellipsis)
            ellipsis_cols: list = get_ellipsis_cols(data, self.target_dtypes)
            for col in ellipsis_cols:
                self.target_dtypes[col] = fill_dtype

    def encode(self, data: DataFrame, /) -> DataFrame:
        return data.astype({k: self.target_dtypes[k] for k in data.columns})

    def decode(self, data: DataFrame, /) -> DataFrame:
        return data.astype({k: self.original_schema[k] for k in data.columns})


def get_ellipsis_cols[T](
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
            case list(cols):
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


@pprint_repr
@dataclass(slots=True, init=False)
class FrameAsDict(FittableEncoder[DataFrame, dict[str, DataFrame]]):
    """Encodes a DataFrame as a dict of DataFrames.

    Note:
        - Each column must be assigned to exactly one group.
        - The special value `...` (`Ellipsis`) can be used to indicate that all unspecified columns belong to a group.
    """

    # Attributes (type hints represent post-fit attributes)
    schema: dict[str, list[str] | EllipsisType]
    r"""The schema for grouping the columns (group-name -> col-name(s))."""

    # Fitted attributes
    original_schema: Mapping[str, Any] = UNDEFINED  # cols -> dtype
    target_schema: Mapping[str, list[str]] = UNDEFINED

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
        self.original_schema = data.dtypes.to_dict()

        self.target_schema = {}
        ellipsis_cols: list[str] = get_ellipsis_cols(data, self.schema.values())
        for group, cols in self.schema.items():
            if cols is ...:  # NOTE: https://github.com/microsoft/pyright/issues/10721
                self.target_schema[group] = ellipsis_cols
            else:
                self.target_schema[group] = cols

        if missing_cols := set().union(*self.target_schema.values()) - set(
            data.columns
        ):
            raise ValueError(f"Missing columns {missing_cols}!")
        if extra_cols := set(data.columns) - set().union(*self.target_schema.values()):
            raise ValueError(f"Extra columns {extra_cols}!")

    def encode(self, data: DataFrame, /) -> dict[str, DataFrame]:
        r"""Encode a DataFrame as a dict of Tensors.

        The encode method ensures treatment of missingness:
        if columns in the dataframe are missing, the correponding tensor columns
        will be filled with `NAN`-values if the datatype allows it.
        """
        return {key: data[cols] for key, cols in self.target_schema.items()}

    def decode(self, data: Mapping[str, DataFrame], /) -> DataFrame:
        # Assemble the DataFrame
        return (
            pd.concat(data.values(), axis="columns")
            .astype(self.original_schema)  # restores dtypes
            .reindex(columns=self.original_schema)  # restores column order
        )
