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
    # Sub-Packages & Modules
    "base",
    "numerical",
    "pytorch",
    "time",
    "transforms",
    # Constants
    "ENCODERS",
    # ABCs & Protocols
    "BackendMixin",
    "BaseEncoder",
    "Encoder",
    "EncoderDict",
    "EncoderList",
    "EncoderProtocol",
    "InvertibleTransform",
    "ParametrizedEncoder",
    "SerializableEncoder",
    "Transform",
    "UniversalEncoder",
    # Classes
    "BoundaryEncoder",
    "BoxCoxEncoder",
    "CSVEncoder",
    "ChainedEncoder",
    "DTypeConverter",
    "DateTimeEncoder",
    "DeepcopyEncoder",
    "DiagonalDecoder",
    "DiagonalEncoder",
    "FrameAsDict",
    "FrameAsTensor",
    "FrameAsTensorDict",
    "FrameEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "JointDecoder",
    "JointEncoder",
    "LinearScaler",
    "LogEncoder",
    "LogitBoxCoxEncoder",
    "LogitEncoder",
    "MappedEncoder",
    "MinMaxScaler",
    "ParallelEncoder",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "PipedEncoder",
    "PositionalEncoder",
    "SocialTimeEncoder",
    "StandardScaler",
    "TensorConcatenator",
    "TensorSplitter",
    "TimeDeltaEncoder",
    "TripletDecoder",
    "TripletEncoder",
    "TupleDecoder",
    "TupleEncoder",
    "WrappedEncoder",
]

from tsdm.encoders import base, numerical, pytorch, time, transforms
from tsdm.encoders.base import (
    BackendMixin,
    BaseEncoder,
    ChainedEncoder,
    DeepcopyEncoder,
    DiagonalDecoder,
    DiagonalEncoder,
    Encoder,
    EncoderDict,
    EncoderList,
    EncoderProtocol,
    IdentityEncoder,
    InverseEncoder,
    InvertibleTransform,
    JointDecoder,
    JointEncoder,
    MappedEncoder,
    ParallelEncoder,
    ParametrizedEncoder,
    PipedEncoder,
    SerializableEncoder,
    Transform,
    TupleDecoder,
    TupleEncoder,
    UniversalEncoder,
    WrappedEncoder,
)
from tsdm.encoders.box_cox import BoxCoxEncoder, LogitBoxCoxEncoder
from tsdm.encoders.dataframe import (
    CSVEncoder,
    DTypeConverter,
    FrameAsDict,
    FrameAsTensor,
    FrameAsTensorDict,
    FrameEncoder,
    TripletDecoder,
    TripletEncoder,
)
from tsdm.encoders.numerical import (
    BoundaryEncoder,
    LinearScaler,
    LogEncoder,
    LogitEncoder,
    MinMaxScaler,
    StandardScaler,
    TensorConcatenator,
    TensorSplitter,
)
from tsdm.encoders.time import (
    DateTimeEncoder,
    PeriodicEncoder,
    PeriodicSocialTimeEncoder,
    PositionalEncoder,
    SocialTimeEncoder,
    TimeDeltaEncoder,
)

ENCODERS: dict[str, type[Encoder]] = {
    "BoundaryEncoder"           : BoundaryEncoder,
    "BoxCoxEncoder"             : BoxCoxEncoder,
    "CSVEncoder"                : CSVEncoder,
    "ChainedEncoder"            : ChainedEncoder,
    "DTypeConverter"            : DTypeConverter,
    "DateTimeEncoder"           : DateTimeEncoder,
    "DeepcopyEncoder"           : DeepcopyEncoder,
    "DiagonalDecoder"           : DiagonalDecoder,
    "DiagonalEncoder"           : DiagonalEncoder,
    "FrameAsDict"               : FrameAsDict,
    "FrameAsTensor"             : FrameAsTensor,
    "FrameAsTensorDict"         : FrameAsTensorDict,
    "FrameEncoder"              : FrameEncoder,
    "IdentityEncoder"           : IdentityEncoder,
    "InverseEncoder"            : InverseEncoder,
    "JointDecoder"              : JointDecoder,
    "JointEncoder"              : JointEncoder,
    "LinearScaler"              : LinearScaler,
    "LogEncoder"                : LogEncoder,
    "LogitBoxCoxEncoder"        : LogitBoxCoxEncoder,
    "LogitEncoder"              : LogitEncoder,
    "MappedEncoder"             : MappedEncoder,
    "MinMaxScaler"              : MinMaxScaler,
    "ParallelEncoder"           : ParallelEncoder,
    "PeriodicEncoder"           : PeriodicEncoder,
    "PeriodicSocialTimeEncoder" : PeriodicSocialTimeEncoder,
    "PipedEncoder"              : PipedEncoder,
    "PositionalEncoder"         : PositionalEncoder,
    "SocialTimeEncoder"         : SocialTimeEncoder,
    "StandardScaler"            : StandardScaler,
    "TensorConcatenator"        : TensorConcatenator,
    "TensorSplitter"            : TensorSplitter,
    "TimeDeltaEncoder"          : TimeDeltaEncoder,
    "TripletDecoder"            : TripletDecoder,
    "TripletEncoder"            : TripletEncoder,
    "TupleDecoder"              : TupleDecoder,
    "TupleEncoder"              : TupleEncoder,
    "WrappedEncoder"            : WrappedEncoder,
}  # fmt: skip
r"""Dictionary of all available encoders."""
