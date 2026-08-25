r"""Encoders for Polars Series and DataFrames."""

__all__ = ["ENCODERS", "FrameEncoder"]

from tsdm.encoders.base import BaseEncoder

from .frame_encoder import FrameEncoder

ENCODERS: dict[str, type[BaseEncoder]] = {
    "FrameEncoder": FrameEncoder,
}
r"""Dictionary of all Polars-specific encoders."""
