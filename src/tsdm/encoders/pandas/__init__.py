r"""Encoders for pandas Series and DataFrame."""

__all__ = [
    "converters",
    "csv",
    "positional",
    "triplet",
    # constants
    "ENCODERS",
    "CSVEncoder",
    "FrameEncoder",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "SocialTimeEncoder",
    "TripletDecoder",
    "TripletEncoder",
]

from . import converters, csv, positional, triplet
from .converters import *  # ruff: ignore[F403]
from .csv import CSVEncoder
from .frame_encoder import FrameEncoder
from .positional import PeriodicEncoder
from .social_time import PeriodicSocialTimeEncoder, SocialTimeEncoder
from .triplet import TripletDecoder, TripletEncoder

__all__ += converters.__all__

ENCODERS = {
    "CSVEncoder"                : CSVEncoder,
    "FrameEncoder"              : FrameEncoder,
    "PeriodicEncoder"           : PeriodicEncoder,
    "PeriodicSocialTimeEncoder" : PeriodicSocialTimeEncoder,
    "SocialTimeEncoder"         : SocialTimeEncoder,
    "TripletDecoder"            : TripletDecoder,
    "TripletEncoder"            : TripletEncoder,
    "FrameAsTensor"             : converters.FrameAsTensor,
    "FrameAsTensorDict"         : converters.FrameAsTensorDict,
    "FrameDTypeConverter"       : converters.FrameDTypeConverter,
    "FrameAsDict"               : converters.FrameAsDict,
}  # fmt: skip
r"""Dictionary of all pandas-specific encoders."""
