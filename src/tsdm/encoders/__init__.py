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
    "torch",
    "universal",
    "time",
    # Constants
    "ID",
    "ENCODERS",
    # ABCs & Protocols
    "SupportsBackend",
    "FittableEncoder",
    "Encoder",
    "EncoderDict",
    "EncoderList",
    "EncoderProtocol",
    "BaseEncoder",
    "ParametrizedEncoder",
    "SupportsSerialization",
    # Classes
    "BoundaryEncoder",
    "BoxCoxEncoder",
    "CSVEncoder",
    "Choice",
    "Compose",
    "DTypeConverter",
    "DateTimeEncoder",
    "DeepcopyEncoder",
    "Diagonal",
    "Duplicate",
    "Fold",
    "Fork",
    "FrameAsDict",
    "FrameAsTensor",
    "FrameAsTensorDict",
    "FrameEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "Meet",
    "LinearScaler",
    "LogEncoder",
    "LogitBoxCoxEncoder",
    "LogitEncoder",
    "MappedEncoder",
    "MinMaxScaler",
    "Parallel",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "Pipe",
    "PositionalEncoder",
    "Replicate",
    "SocialTimeEncoder",
    "StandardScaler",
    "TensorConcatenator",
    "TensorSplitter",
    "TimeDeltaEncoder",
    "TripletDecoder",
    "TripletEncoder",
    "TupleUnwrapper",
    "TupleWrapper",
    "WrappedEncoder",
    # Functions
    "choice",
    "compose",
    "duplicate",
    "fold",
    "fork",
    "invert",
    "map_encoders",
    "parallel",
    "pipe",
    "repeat",
    "replicate",
    "wrap",
]

from tsdm.encoders import time, torch, universal
from tsdm.encoders.base import (
    ID,
    BaseEncoder,
    # constants
    Choice,
    Compose,
    DeepcopyEncoder,
    Diagonal,
    Duplicate,
    Encoder,
    EncoderDict,
    EncoderList,
    EncoderProtocol,
    FittableEncoder,
    Fold,
    Fork,
    IdentityEncoder,
    InverseEncoder,
    MappedEncoder,
    Meet,
    Parallel,
    ParametrizedEncoder,
    Pipe,
    Replicate,
    TupleUnwrapper,
    TupleWrapper,
    WrappedEncoder,
    choice,
    # functions
    compose,
    duplicate,
    fold,
    fork,
    invert,
    map_encoders,
    parallel,
    pipe,
    repeat,
    replicate,
    wrap,
)
from tsdm.encoders.box_cox import (
    BoxCoxEncoder,
    LogEncoder,
    LogitBoxCoxEncoder,
    LogitEncoder,
)
from tsdm.encoders.pandas import (
    CSVEncoder,
    DTypeConverter,
    FrameAsDict,
    FrameAsTensor,
    FrameAsTensorDict,
    FrameEncoder,
    TripletDecoder,
    TripletEncoder,
)
from tsdm.encoders.protocols import SupportsBackend, SupportsSerialization
from tsdm.encoders.time import (
    PeriodicEncoder,
    PeriodicSocialTimeEncoder,
    PositionalEncoder,
    SocialTimeEncoder,
)
from tsdm.encoders.universal import (
    BoundaryEncoder,
    LinearScaler,
    MinMaxScaler,
    StandardScaler,
    TensorConcatenator,
    TensorSplitter,
)
from tsdm.encoders.universal.temporal import DateTimeEncoder, TimeDeltaEncoder

ENCODERS: dict[str, type[BaseEncoder]] = {
    "BoundaryEncoder"           : BoundaryEncoder,
    "BoxCoxEncoder"             : BoxCoxEncoder,
    "CSVEncoder"                : CSVEncoder,
    "Choice"                    : Choice,
    "Compose"                   : Compose,
    "DTypeConverter"            : DTypeConverter,
    "DateTimeEncoder"           : DateTimeEncoder,
    "DeepcopyEncoder"           : DeepcopyEncoder,
    "Diagonal"                  : Diagonal,
    "Duplicate"                 : Duplicate,
    "Fold"                      : Fold,
    "Fork"                      : Fork,
    "FrameAsDict"               : FrameAsDict,
    "FrameAsTensor"             : FrameAsTensor,
    "FrameAsTensorDict"         : FrameAsTensorDict,
    "FrameEncoder"              : FrameEncoder,
    "IdentityEncoder"           : IdentityEncoder,
    "InverseEncoder"            : InverseEncoder,
    "LinearScaler"              : LinearScaler,
    "LogEncoder"                : LogEncoder,
    "LogitBoxCoxEncoder"        : LogitBoxCoxEncoder,
    "LogitEncoder"              : LogitEncoder,
    "MappedEncoder"             : MappedEncoder,
    "Meet"                      : Meet,
    "MinMaxScaler"              : MinMaxScaler,
    "Parallel"                  : Parallel,
    "PeriodicEncoder"           : PeriodicEncoder,
    "PeriodicSocialTimeEncoder" : PeriodicSocialTimeEncoder,
    "Pipe"                      : Pipe,
    "PositionalEncoder"         : PositionalEncoder,
    "Replicate"                 : Replicate,
    "SocialTimeEncoder"         : SocialTimeEncoder,
    "StandardScaler"            : StandardScaler,
    "TensorConcatenator"        : TensorConcatenator,
    "TensorSplitter"            : TensorSplitter,
    "TimeDeltaEncoder"          : TimeDeltaEncoder,
    "TripletDecoder"            : TripletDecoder,
    "TripletEncoder"            : TripletEncoder,
    "TupleUnwrapper"            : TupleUnwrapper,
    "TupleWrapper"              : TupleWrapper,
    "WrappedEncoder"            : WrappedEncoder,
}  # fmt: skip
r"""Dictionary of all available encoders."""
