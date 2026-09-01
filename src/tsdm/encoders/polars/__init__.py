r"""Encoders for Polars Series and DataFrames."""

__all__ = ["ENCODERS", "FrameEncoder"]


from .frame_encoder import FrameEncoder

ENCODERS = {
    "FrameEncoder": FrameEncoder,
}
r"""Dictionary of all Polars-specific encoders."""
