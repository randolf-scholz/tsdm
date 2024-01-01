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

Note:
    Some important unit kinds:

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
        - Encoders: `BoxCoxEncoder`
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
        - support: `{1, ..., K}`
        - dimensionless: `True`
        - ordered: `False`
        - Encoders: `OneHotEncoder`
    - ordinal
        - support: `{1, ..., K}`
        - dimensionless: `True`
        - ordered: `True`
        - Encoders: `OrdinalEncoder`, `PositionalEncoder`
    - cyclic
        - support: $[0, 2π)$
        - dimensionless: `True`
        - ordered: `False`
        - Encoders: `SinusoidalEncoder`, `CosineEncoder`, `PeriodicEncoder`

See Also:
    - `tsdm.encoders.functional` for functional implementations.
    - `tsdm.encoders` for modular implementations.
"""

#  TODO: Add more encoders
# - Target Encoding: enc(x) = mean(enc(y|x))
# - Binary Encoding: enx(x) = ...
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
    "transforms",
    # Modules
    "base",
    "numerical",
    "time",
    "pytorch",
    # Types
    "Encoder",
    "EncoderProtocol",
    "InvertibleTransform",
    "Transform",
    # Constants
    "ENCODERS",
    # ABC
    "BaseEncoder",
    # Classes
    "BoundaryEncoder",
    "BoxCoxEncoder",
    "ChainedEncoder",
    "CloneEncoder",
    "CopyEncoder",
    "OldDateTimeEncoder",
    "DuplicateEncoder",
    "FloatEncoder",
    "FrameAsDict",
    "FrameAsTuple",
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
    "ParallelEncoder",
    "SocialTimeEncoder",
    "StandardScaler",
    "TensorConcatenator",
    "TensorEncoder",
    "TensorSplitter",
    "Time2Float",
    "TimeDeltaEncoder",
    "TripletDecoder",
    "TripletEncoder",
    "ValueEncoder",
    # Aliases
]

from tsdm.encoders import base, numerical, pytorch, time, transforms
from tsdm.encoders.base import (
    BaseEncoder,
    ChainedEncoder,
    CloneEncoder,
    CopyEncoder,
    DuplicateEncoder,
    Encoder,
    EncoderProtocol,
    IdentityEncoder,
    InvertibleTransform,
    MappingEncoder,
    ParallelEncoder,
    Transform,
)
from tsdm.encoders.box_cox import BoxCoxEncoder, LogitBoxCoxEncoder
from tsdm.encoders.dataframe import (
    FrameAsDict,
    FrameAsTuple,
    FrameEncoder,
    FrameIndexer,
    FrameSplitter,
    TensorEncoder,
    TripletDecoder,
    TripletEncoder,
    ValueEncoder,
)
from tsdm.encoders.numerical import (
    BoundaryEncoder,
    FloatEncoder,
    IntEncoder,
    LinearScaler,
    LogEncoder,
    LogitEncoder,
    MinMaxScaler,
    StandardScaler,
    TensorConcatenator,
    TensorSplitter,
)
from tsdm.encoders.time import (
    OldDateTimeEncoder,
    PeriodicEncoder,
    PeriodicSocialTimeEncoder,
    PositionalEncoder,
    SocialTimeEncoder,
    Time2Float,
    TimeDeltaEncoder,
)

ENCODERS: dict[str, type[Encoder]] = {
    # "BaseEncoder": BaseEncoder,  # Only concrete classes!
    "BoundaryEncoder": BoundaryEncoder,
    "CopyEncoder": CopyEncoder,
    "BoxCoxEncoder": BoxCoxEncoder,
    "ChainedEncoder": ChainedEncoder,
    "CloneEncoder": CloneEncoder,
    "DateTimeEncoder": OldDateTimeEncoder,
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
    "ProductEncoder": ParallelEncoder,
    "SocialTimeEncoder": SocialTimeEncoder,
    "StandardScaler": StandardScaler,
    "TensorConcatenator": TensorConcatenator,
    "TensorEncoder": TensorEncoder,
    "TensorSplitter": TensorSplitter,
    "Time2Float": Time2Float,
    "TimeDeltaEncoder": TimeDeltaEncoder,
    "TripletEncoder": TripletEncoder,
}
r"""Dictionary of all available encoders."""
