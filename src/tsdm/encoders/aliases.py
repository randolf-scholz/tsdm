"""Name aliases for encoders."""

__all__ = [
    "Standardizer",
    "Frame2TensorDict",
]

from tsdm.encoders.dataframe import FrameAsDict
from tsdm.encoders.numerical import StandardScaler

Standardizer = StandardScaler
Frame2TensorDict = FrameAsDict  # Do not remove! For old pickle files
"""Alias for `FrameAsDict`"""
