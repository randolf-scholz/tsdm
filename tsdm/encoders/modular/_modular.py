r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    "BaseEncoder",
    "Time2Float",
]

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from pandas import Series

__logger__ = logging.getLogger(__name__)


class BaseEncoder(ABC):
    """Base class that all encoders must subclass."""

    @abstractmethod
    def fit(self, data):
        r"""Fit parameters to given data."""

    @abstractmethod
    def encode(self, data):
        r"""Transform the data."""

    @abstractmethod
    def decode(self, data):
        r"""Reverse the applied transformation."""

    transform = encode
    fit_transform = encode
    inverse_transform = decode


class Time2Float(BaseEncoder):
    r"""Convert ``Series`` encoded as ``datetime64`` or ``timedelta64`` to ``floating``.

    By default, the data is mapped onto the unit interval `[0,1]`

    Parameters
    ----------
    ds: Series

    Returns
    -------
    Series
    """

    original_dtype: np.dtype
    offset: Any
    common_interval: Any
    scale: Any

    def __init__(self, normalization: Literal["gcd", "max", "none"] = "max"):
        """Choose the normalizations scheme.

        Parameters
        ----------
        normalization: Literal["gcd", "max", "none"], default="max"
        """
        self.normalization = normalization

    def fit(self, ds: Series):
        r"""Fit to the data.

        Parameters
        ----------
        ds: Series
        """
        self.original_dtype = ds.dtype
        self.offset = ds[0].copy()
        assert (
            ds.is_monotonic_increasing
        ), "Time-Values must be monotonically increasing!"

        assert not (
            np.issubdtype(self.original_dtype, np.floating)
            and self.normalization == "gcd"
        ), f"{self.normalization=} illegal when original dtype is floating."

        if np.issubdtype(ds.dtype, np.datetime64):
            ds = ds.view("datetime64[ns]")
            self.offset = ds[0].copy()
            timedeltas = ds - self.offset
        elif np.issubdtype(ds.dtype, np.timedelta64):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.integer):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.floating):
            __logger__.warning("Array is already floating dtype.")
            timedeltas = ds
        else:
            raise ValueError(f"{ds.dtype=} not supported")

        if self.normalization == "none":
            self.scale = np.array(1).view("timedelta64[ns]")
        elif self.normalization == "gcd":
            self.scale = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")
        elif self.normalization == "max":
            self.scale = ds[-1]
        else:
            raise ValueError(f"{self.normalization=} not understood")

    def encode(self, ds: Series) -> Series:
        """Encode the data.

        Roughly equal to::
            ds ⟶ ((ds - offset)/scale).astype(float)

        Parameters
        ----------
        ds: Series

        Returns
        -------
        Series
        """
        if np.issubdtype(ds.dtype, np.integer):
            return ds
        if np.issubdtype(ds.dtype, np.datetime64):
            ds = ds.view("datetime64[ns]")
            self.offset = ds[0].copy()
            timedeltas = ds - self.offset
        elif np.issubdtype(ds.dtype, np.timedelta64):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.floating):
            __logger__.warning("Array is already floating dtype.")
            return ds
        else:
            raise ValueError(f"{ds.dtype=} not supported")

        common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

        return (timedeltas / common_interval).astype(float)

    def decode(self, ds: Series) -> Series:
        """Apply the inverse transformation.

        Roughly equal to:
        ``ds ⟶ (scale*ds + offset).astype(original_dtype)``
        """


# class DataFrameEncoder(BaseEncoder):
#     r"""Combine multiple encoders into a single one."""
#
#     def __init__(
#         self,
#         index_encoder: BaseEncoder,
#         column_encoders: dict[Union[str, tuple[str, ...]], BaseEncoder],
#     ):
#         """
#
#         Parameters
#         ----------
#         index_encoder
#         column_encoders
#         """
#         self.index_encoder = index_encoder
#         self.column_encoders = column_encoders
#
#     def fit(self, df: DataFrame):
#         """
#
#         Parameters
#         ----------
#         df
#         """
#
#     def encode(self, df: DataFrame) -> DataFrame:
#         ...
#
#     def decode(self, df: DataFrame) -> DataFrame:
#         ...
