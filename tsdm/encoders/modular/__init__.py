r"""Implementation of encoders.

Notes
-----
Contains encoders in modular form.
  - See :mod:`tsdm.encoders.functional` for functional implementations.
"""

__all__ = [
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
    "IdentityEncoder",
    "IntEncoder",
    "MinMaxScaler",
    "PositionalEncoder",
    "ProductEncoder",
    "Standardizer",
    "TensorEncoder",
    "Time2Float",
    "TripletEncoder",
]

import logging
from typing import Final

from sklearn import preprocessing as sk_preprocessing
from sklearn.base import BaseEstimator

from tsdm.encoders.modular._modular import (
    BaseEncoder,
    ChainedEncoder,
    ConcatEncoder,
    DataFrameEncoder,
    DateTimeEncoder,
    FloatEncoder,
    FrameEncoder,
    FrameSplitter,
    IdentityEncoder,
    IntEncoder,
    MinMaxScaler,
    PositionalEncoder,
    ProductEncoder,
    Standardizer,
    TensorEncoder,
    Time2Float,
    TripletEncoder,
)
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)


SklearnModularEncoders: Final[LookupTable[type[BaseEstimator]]] = {
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


ModularEncoders: Final[LookupTable[type[BaseEstimator]]] = {
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
