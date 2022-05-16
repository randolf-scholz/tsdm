r"""Implementation of encoders.

Notes
-----
Contains encoders in modular form.
  - See `tsdm.encoders.functional` for functional implementations.
"""

__all__ = [
    # Modules
    "generic",
    "numerical",
    # Constants
    "ModularEncoder",
    "ModularEncoders",
    # ABC
    "BaseEncoder",
    # Classes
    "ChainedEncoder",
    "ConcatEncoder",
    "DataFrameEncoder",
    "DateTimeEncoder",
    "FloatEncoder",
    "FrameEncoder",
    "FrameSplitter",
    "FrameIndexer",
    "IdentityEncoder",
    "IntEncoder",
    "PositionalEncoder",
    "ProductEncoder",
    "TensorEncoder",
    "Time2Float",
    "TimeDeltaEncoder",
    "TripletEncoder",
    "LogEncoder",
    "MinMaxScaler",
    "Standardizer",
    "ValueEncoder",
    "DuplicateEncoder",
    "CloneEncoder",
    "PeriodicEncoder",
    "SocialTimeEncoder",
    "PeriodicSocialTimeEncoder",
    "TripletDecoder",
]

import logging
from typing import Final

from sklearn import preprocessing as sk_preprocessing
from sklearn.base import BaseEstimator

from tsdm.encoders.modular import generic, numerical
from tsdm.encoders.modular._dtypes import (
    DateTimeEncoder,
    FloatEncoder,
    IntEncoder,
    Time2Float,
    TimeDeltaEncoder,
)
from tsdm.encoders.modular._modular import (
    ConcatEncoder,
    DataFrameEncoder,
    FrameEncoder,
    FrameIndexer,
    FrameSplitter,
    PeriodicEncoder,
    PeriodicSocialTimeEncoder,
    PositionalEncoder,
    SocialTimeEncoder,
    TensorEncoder,
    TripletDecoder,
    TripletEncoder,
    ValueEncoder,
)
from tsdm.encoders.modular.generic import (
    BaseEncoder,
    ChainedEncoder,
    CloneEncoder,
    DuplicateEncoder,
    IdentityEncoder,
    ProductEncoder,
)
from tsdm.encoders.modular.numerical import LogEncoder, MinMaxScaler, Standardizer

__logger__ = logging.getLogger(__name__)


SklearnModularEncoders: Final[dict[str, type[BaseEstimator]]] = {
    "Binarizer": sk_preprocessing.Binarizer,
    "FunctionTransformer": sk_preprocessing.FunctionTransformer,
    "KBinsDiscretizer": sk_preprocessing.KBinsDiscretizer,
    "KernelCenterer": sk_preprocessing.KernelCenterer,
    "LabelBinarizer": sk_preprocessing.LabelBinarizer,
    "LabelEncoder": sk_preprocessing.LabelEncoder,
    "MaxAbsScaler": sk_preprocessing.MaxAbsScaler,
    "MinMaxScaler": sk_preprocessing.MinMaxScaler,
    "MultiLabelBinarizer": sk_preprocessing.MultiLabelBinarizer,
    "Normalizer": sk_preprocessing.Normalizer,
    "OneHotEncoder": sk_preprocessing.OneHotEncoder,
    "OrdinalEncoder": sk_preprocessing.OrdinalEncoder,
    "PolynomialFeatures": sk_preprocessing.PolynomialFeatures,
    "PowerTransformer": sk_preprocessing.PowerTransformer,
    "QuantileTransformer": sk_preprocessing.QuantileTransformer,
    "RobustScaler": sk_preprocessing.RobustScaler,
    "SplineTransformer": sk_preprocessing.SplineTransformer,
    "StandardScaler": sk_preprocessing.StandardScaler,
}
r"""Dictionary of all available sklearn encoders."""

ModularEncoder = BaseEncoder
r"""Type hint for modular encoders."""


ModularEncoders: Final[dict[str, type[BaseEstimator]]] = {
    "ChainedEncoder": ChainedEncoder,
    "DataFrameEncoder": DataFrameEncoder,
    "DateTimeEncoder": DateTimeEncoder,
    "FloatEncoder": FloatEncoder,
    "IdentityEncoder": IdentityEncoder,
    "MinMaxScaler": MinMaxScaler,
    "Standardizer": Standardizer,
    "TensorEncoder": TensorEncoder,
    "Time2Float": Time2Float,
    "IntEncoder": IntEncoder,
    "TripletEncoder": TripletEncoder,
    "ConcatEncoder": ConcatEncoder,
}
r"""Dictionary of all available modular encoders."""
