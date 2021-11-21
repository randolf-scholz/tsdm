r"""Implementation of encoders.

Notes
-----
Contains encoders in modular form.
  - See :mod:`tsdm.encoders.functional` for functional implementations.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "ModularEncoder",
    "ModularEncoders",
    # Classes
    "BaseEncoder",
    "ChainedEncoder",
    "DataFrameEncoder",
    "DateTimeEncoder",
    "FloatEncoder",
    "IdentityEncoder",
    "MinMaxScaler",
    "Standardizer",
    "TensorEncoder",
    "Time2Float",
]

import logging
from typing import Final

from sklearn import preprocessing as sk_preprocessing
from sklearn.base import BaseEstimator

from tsdm.encoders.modular._modular import (
    BaseEncoder,
    ChainedEncoder,
    DataFrameEncoder,
    DateTimeEncoder,
    FloatEncoder,
    IdentityEncoder,
    MinMaxScaler,
    Standardizer,
    TensorEncoder,
    Time2Float,
)
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

ModularEncoder = BaseEstimator
r"""Type hint for modular encoders."""

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

ModularEncoders: Final[LookupTable[type[BaseEstimator]]] = SklearnModularEncoders
r"""Dictionary of all available modular encoders."""
