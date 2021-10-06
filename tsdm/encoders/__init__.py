r"""Encoders for different types of data.

- Target Encoding: enc(x) = mean(enc(y|x))
- Binary Encoding: enx(x) =
- Hash Encoder: enc(x) = binary(hash(x))
- Effect/Sum/Deviation Encoding:
- Sum Encoding
- ECC Binary Encoding:
- Ordinal Coding: (cᵢ | i=1:n) -> (i| i=1...n)
- Dummy Encoding: like one-hot, with (0,...,0) added as a category
- word2vec
- Learned encoding:


Hierarchical Categoricals:
- Sum Coding
- Helmert Coding
- Polynomial Coding
- Backward Difference Coding:
"""

from __future__ import annotations

__all__ = [
    # Constants
    "FunctionalEncoder",
    "FunctionalEncoders",
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
from typing import Any, Final, Union

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

from tsdm.encoders.functional import FunctionalEncoder, FunctionalEncoders

LOGGER = logging.getLogger(__name__)


ModularEncoder = Any
r"""Type hint for encoders."""

SklearnModularEncoders: Final[dict[str, ModularEncoder]] = {
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
r"""Dictionary of all available encoders."""

ModularEncoders = SklearnModularEncoders

Encoder = Union[ModularEncoder, FunctionalEncoder]
Encoders: Final[dict[str, Encoder]] = ModularEncoders | FunctionalEncoders
