r"""Implementation of Encoders.

Role & Specification
--------------------

Encoders are used in multiple contexts
  - Perform preprocessing for task objects: For example, a task might ask to evaluate on
    standardized features. In this case, an pre_encoder object is associated with the task that
    will perform this preprocessing at task creation time.
  - Perform data encoding tasks such as encoding of categorical variables.
  - Transform data from one framework to another, like `numpy` → `torch`

Specification:
  - Encoders **must** be reversible.
  - Modules that are not reversible, we call transformations.
      - Example: Convert logit output of a NN to a class prediction.

Notes
-----
Contains encoders in both modular and functional form.
  - See `tsdm.encoders.functional` for functional implementations.
  - See `tsdm.encoders` for modular implementations.
"""
#  TODO:
# - Target Encoding: enc(x) = mean(enc(y|x))
# - Binary Encoding: enx(x) =
# - Hash Encoder: enc(x) = binary(hash(x))
# - Effect/Sum/Deviation Encoding:
# - Sum Encoding
# - ECC Binary Encoding:
# - Ordinal Coding: (cᵢ | i=1:n) -> (i| i=1...n)
# - Dummy Encoding: like one-hot, with (0,...,0) added as a category
# - word2vec
# - Learned encoding:
#
# Hierarchical Categoricals:
# - Sum Coding
# - Helmert Coding
# - Polynomial Coding
# - Backward Difference Coding:

__all__ = [
    # Sub-Packages
    "functional",
    # Modules
    "base",
    "numerical",
    "time",
    "torch",
    # Constants
    "Encoder",
    "ENCODERS",
    "ModularEncoder",
    "ModularEncoders",
    "FunctionalEncoder",
    "FunctionalEncoders",
    # ABC
    "BaseEncoder",
    # Classes
    "ChainedEncoder",
    "CloneEncoder",
    "ConcatEncoder",
    "DataFrameEncoder",
    "DateTimeEncoder",
    "DuplicateEncoder",
    "FrameEncoder",
    "FrameIndexer",
    "FrameSplitter",
    "IdentityEncoder",
    "LogEncoder",
    "MinMaxScaler",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "PositionalEncoder",
    "ProductEncoder",
    "SocialTimeEncoder",
    "Standardizer",
    "TensorEncoder",
    "Time2Float",
    "TimeDeltaEncoder",
    "TripletDecoder",
    "TripletEncoder",
    "ValueEncoder",
]

from typing import Final, Union

from sklearn import preprocessing as sk_preprocessing
from sklearn.base import BaseEstimator

from tsdm.encoders import base, functional, numerical, time, torch
from tsdm.encoders._modular import (
    ConcatEncoder,
    DataFrameEncoder,
    FrameEncoder,
    FrameIndexer,
    FrameSplitter,
    PositionalEncoder,
    TensorEncoder,
    TripletDecoder,
    TripletEncoder,
    ValueEncoder,
)
from tsdm.encoders.base import (
    BaseEncoder,
    ChainedEncoder,
    CloneEncoder,
    DuplicateEncoder,
    IdentityEncoder,
    ProductEncoder,
)
from tsdm.encoders.functional import FunctionalEncoder, FunctionalEncoders
from tsdm.encoders.numerical import (
    FloatEncoder,
    IntEncoder,
    LogEncoder,
    MinMaxScaler,
    Standardizer,
)
from tsdm.encoders.time import (
    DateTimeEncoder,
    PeriodicEncoder,
    PeriodicSocialTimeEncoder,
    SocialTimeEncoder,
    Time2Float,
    TimeDeltaEncoder,
)

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


Encoder = Union[FunctionalEncoder, ModularEncoder]
r"""Type hint for encoders."""

ENCODERS: Final[dict[str, Union[FunctionalEncoder, type[ModularEncoder]]]] = {
    **ModularEncoders,
    **FunctionalEncoders,
}
r"""Dictionary of all available encoders."""


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
