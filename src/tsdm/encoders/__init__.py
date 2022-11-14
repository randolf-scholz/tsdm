r"""Implementation of Encoders.

Role & Specification
--------------------

Encoders are used in multiple contexts
  - Perform preprocessing for task objects: For example, a task might ask to evaluate on
    standardized features. In this case, a pre_encoder object is associated with the task that
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


Some important units

- numerical
    - finite
        - support: {a₁, ..., aₙ}
        - dimensionless: False
        - ordered: True
    - discrete:
        - support: ℤ
        - dimensionless: True
        - ordered: True
    - absolute:
        - support: [0, +∞)
        - dimensionless: False
        - ordered: True
        - Encoders: `BoxCox`
    - factor:
        - support: (0, +∞)
        - dimensionless: True
        - ordered: True
        - Encoders: `Logarithm`
    - percent
        - support: [0, 100]
        - dimensionless: True
        - ordered: True
        - Encoders: `Logit @ MinMaxScalar[0, 100]`, `HardLogit @ MinMaxScalar[0, 100]`
    - bounded
        - support: [a, b]
        - dimensionless: False
        - ordered: True
        - Encoders: `Logit @ MinMaxScalar[a, b]`, `HardLogit @ MinMaxScalar[a, b]`
    - linear
        - support: (-∞, +∞)
        - dimensionless: False
        - ordered: True
        - Encoders: `StandardScalar` (diagonal or full covariance)
- category
    - support {1, ..., K}
    - dimensionless: True
    - ordered: False
    - Encoders: `OneHotEncoder`
- ordinal
    - support {1, ..., K}
    - dimensionless: True
    - ordered: True
    - Encoders: `OrdinalEncoder`, `PositionalEncoder`
- cyclic
    - support: [0, 2π)
    - dimensionless: True
    - ordered: False
    - Encoders: `SinusoidalEncoder`, `CosineEncoder`, `PeriodicEncoder`
"""
#  TODO: Add more encoders
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
    "pytorch",
    # Types
    "Encoder",
    "ModularEncoder",
    "FunctionalEncoder",
    # Constants
    "ENCODERS",
    "MODULAR_ENCODERS",
    "FUNCTIONAL_ENCODERS",
    "SKLEARN_MODULAR_ENCODERS",
    "SKLEARN_FUNCTIONAL_ENCODERS",
    # ABC
    "BaseEncoder",
    # Classes
    "BoundaryEncoder",
    "BoxCoxEncoder",
    "ChainedEncoder",
    "CloneEncoder",
    "DateTimeEncoder",
    "DuplicateEncoder",
    "FloatEncoder",
    "Frame2TensorDict",
    "FrameEncoder",
    "FrameIndexer",
    "FrameSplitter",
    "IdentityEncoder",
    "IntEncoder",
    "LinearScaler",
    "LogEncoder",
    "LogitBoxCoxEncoder",
    "LogitEncoder",
    "MappingEncoder",
    "MinMaxScaler",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "PositionalEncoder",
    "ProductEncoder",
    "SocialTimeEncoder",
    "Standardizer",
    "TensorConcatenator",
    "TensorEncoder",
    "TensorSplitter",
    "Time2Float",
    "TimeDeltaEncoder",
    "TimeSlicer",
    "TripletDecoder",
    "TripletEncoder",
    "ValueEncoder",
]

from typing import Final, TypeAlias

from sklearn import preprocessing as sk_preprocessing
from sklearn.base import BaseEstimator

from tsdm.encoders import base, functional, numerical, pytorch, time
from tsdm.encoders._deprecated import DataFrameEncoder
from tsdm.encoders.base import (
    BaseEncoder,
    ChainedEncoder,
    CloneEncoder,
    DuplicateEncoder,
    IdentityEncoder,
    MappingEncoder,
    ProductEncoder,
)
from tsdm.encoders.box_cox import BoxCoxEncoder, LogitBoxCoxEncoder
from tsdm.encoders.dataframe import (
    Frame2TensorDict,
    FrameEncoder,
    FrameIndexer,
    FrameSplitter,
    TensorEncoder,
    TripletDecoder,
    TripletEncoder,
    ValueEncoder,
)
from tsdm.encoders.functional import (
    FUNCTIONAL_ENCODERS,
    SKLEARN_FUNCTIONAL_ENCODERS,
    FunctionalEncoder,
)
from tsdm.encoders.numerical import (
    BoundaryEncoder,
    FloatEncoder,
    IntEncoder,
    LinearScaler,
    LogEncoder,
    LogitEncoder,
    MinMaxScaler,
    Standardizer,
    TensorConcatenator,
    TensorSplitter,
)
from tsdm.encoders.time import (
    DateTimeEncoder,
    PeriodicEncoder,
    PeriodicSocialTimeEncoder,
    PositionalEncoder,
    SocialTimeEncoder,
    Time2Float,
    TimeDeltaEncoder,
    TimeSlicer,
)

ModularEncoder: TypeAlias = BaseEncoder
r"""Type hint for modular encoders."""

Encoder: TypeAlias = FunctionalEncoder | ModularEncoder
r"""Type hint for encoders."""

SKLEARN_MODULAR_ENCODERS: Final[dict[str, type[BaseEstimator]]] = {
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

MODULAR_ENCODERS: Final[dict[str, type[BaseEstimator]]] = {
    "BaseEncoder": BaseEncoder,
    "BoundaryEncoder": BoundaryEncoder,
    "BoxCoxEncoder": BoxCoxEncoder,
    "ChainedEncoder": ChainedEncoder,
    "CloneEncoder": CloneEncoder,
    "DataFrameEncoder": DataFrameEncoder,
    "DateTimeEncoder": DateTimeEncoder,
    "DuplicateEncoder": DuplicateEncoder,
    "FloatEncoder": FloatEncoder,
    "IdentityEncoder": IdentityEncoder,
    "IntEncoder": IntEncoder,
    "LinearScaler": LinearScaler,
    "LogEncoder": LogEncoder,
    "LogitEncoder": LogitEncoder,
    "LogitBoxCoxEncoder": LogitBoxCoxEncoder,
    "MappingEncoder": MappingEncoder,
    "MinMaxScaler": MinMaxScaler,
    "PeriodicEncoder": PeriodicEncoder,
    "PeriodicSocialTimeEncoder": PeriodicSocialTimeEncoder,
    "ProductEncoder": ProductEncoder,
    "SocialTimeEncoder": SocialTimeEncoder,
    "Standardizer": Standardizer,
    "TensorConcatenator": TensorConcatenator,
    "TensorEncoder": TensorEncoder,
    "TensorSplitter": TensorSplitter,
    "Time2Float": Time2Float,
    "TimeDeltaEncoder": TimeDeltaEncoder,
    "TripletEncoder": TripletEncoder,
}
r"""Dictionary of all available modular encoders."""


ENCODERS: Final[dict[str, FunctionalEncoder | type[ModularEncoder]]] = {
    **FUNCTIONAL_ENCODERS,
    **MODULAR_ENCODERS,
}
r"""Dictionary of all available encoders."""

del Final, TypeAlias, sk_preprocessing, BaseEstimator
