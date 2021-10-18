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
    "Time2Float",
    # Classes - Sklearn
    "Binarizer",
    "FunctionTransformer",
    "KBinsDiscretizer",
    "KernelCenterer",
    "LabelBinarizer",
    "LabelEncoder",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MultiLabelBinarizer",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RobustScaler",
    "SplineTransformer",
    "StandardScaler",
]

import logging
from typing import Final

from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    KernelCenterer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MultiLabelBinarizer,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)

from tsdm.encoders.modular._modular import BaseEncoder, Time2Float
from tsdm.util.types import LookupTable

LOGGER = logging.getLogger(__name__)

ModularEncoder = BaseEstimator
r"""Type hint for modular encoders."""

SklearnModularEncoders: Final[LookupTable[type[BaseEstimator]]] = {
    "Binarizer": Binarizer,
    "FunctionTransformer": FunctionTransformer,
    "KBinsDiscretizer": KBinsDiscretizer,
    "KernelCenterer": KernelCenterer,
    "LabelBinarizer": LabelBinarizer,
    "LabelEncoder": LabelEncoder,
    "MaxAbsScaler": MaxAbsScaler,
    "MinMaxScaler": MinMaxScaler,
    "MultiLabelBinarizer": MultiLabelBinarizer,
    "Normalizer": Normalizer,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
    "PolynomialFeatures": PolynomialFeatures,
    "PowerTransformer": PowerTransformer,
    "QuantileTransformer": QuantileTransformer,
    "RobustScaler": RobustScaler,
    "SplineTransformer": SplineTransformer,
    "StandardScaler": StandardScaler,
}
r"""Dictionary of all available sklearn encoders."""

ModularEncoders: Final[LookupTable[type[BaseEstimator]]] = SklearnModularEncoders
r"""Dictionary of all available modular encoders."""
