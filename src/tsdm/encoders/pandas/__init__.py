r"""Encoders for pandas Series and DataFrame."""

__all__ = [
    "ENCODERS",
    "CSVEncoder",
    "FrameEncoder",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "SocialTimeEncoder",
    "TripletDecoder",
    "TripletEncoder",
]

from tsdm.encoders.base import BaseEncoder

from .csv import CSVEncoder
from .frame_encoder import FrameEncoder
from .positional import PeriodicEncoder
from .social_time import PeriodicSocialTimeEncoder, SocialTimeEncoder
from .triplet import TripletDecoder, TripletEncoder

ENCODERS: dict[str, type[BaseEncoder]] = {
    "CSVEncoder"                : CSVEncoder,
    "FrameEncoder"              : FrameEncoder,
    "PeriodicEncoder"           : PeriodicEncoder,
    "PeriodicSocialTimeEncoder" : PeriodicSocialTimeEncoder,
    "SocialTimeEncoder"         : SocialTimeEncoder,
    "TripletDecoder"            : TripletDecoder,
    "TripletEncoder"            : TripletEncoder,
}  # fmt: skip
r"""Dictionary of all pandas-specific encoders."""
