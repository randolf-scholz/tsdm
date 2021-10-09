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
    "ModularEncoderType",
    # Classes
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

LOGGER = logging.getLogger(__name__)

ModularEncoder = BaseEstimator
r"""Type hint for modular encoders."""

ModularEncoderType = type[BaseEstimator]
r"""Type hint for modular encoders."""

SklearnModularEncoders: Final[dict[str, ModularEncoderType]] = {
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

ModularEncoders = SklearnModularEncoders
r"""Dictionary of all available modular encoders."""
