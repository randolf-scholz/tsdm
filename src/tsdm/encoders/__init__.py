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
    "positional",
    "universal",
    "torch",
    "pandas",
    "polars",
    # Constants
    "ID",
    "ENCODERS",
    # ABCs & Protocols
    "FittableEncoder",
    "Encoder",
    "EncoderDict",
    "EncoderList",
    "EncoderProtocol",
    "BaseEncoder",
    "ParametrizedEncoder",
    # Classes
    "BoundaryEncoder",
    "BoxCoxEncoder",
    "Choice",
    "Compose",
    "FrameDTypeConverter",
    "DateTimeEncoder",
    "DeepcopyEncoder",
    "Diagonal",
    "Duplicate",
    "Fold",
    "Fork",
    "FrameAsDict",
    "FrameAsTensor",
    "FrameAsTensorDict",
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
    "Pipe",
    "PositionalEncoder",
    "Replicate",
    "StandardScaler",
    "TensorConcatenator",
    "TensorSplitter",
    "TimeDeltaEncoder",
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

from . import pandas, polars, positional, torch, universal
from .base import (
    ID,
    BaseEncoder,
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
from .box_cox import BoxCoxEncoder, LogEncoder, LogitBoxCoxEncoder, LogitEncoder
from .converters import (
    FrameAsDict,
    FrameAsTensor,
    FrameAsTensorDict,
    FrameDTypeConverter,
)
from .positional import PositionalEncoder
from .universal import (
    BoundaryEncoder,
    DateTimeEncoder,
    LinearScaler,
    MinMaxScaler,
    StandardScaler,
    TensorConcatenator,
    TensorSplitter,
    TimeDeltaEncoder,
)

ENCODERS: dict[str, type[BaseEncoder]] = {
    "BoundaryEncoder"           : BoundaryEncoder,
    "BoxCoxEncoder"             : BoxCoxEncoder,
    "Choice"                    : Choice,
    "Compose"                   : Compose,
    "FrameDTypeConverter"       : FrameDTypeConverter,
    "DateTimeEncoder"           : DateTimeEncoder,
    "DeepcopyEncoder"           : DeepcopyEncoder,
    "Diagonal"                  : Diagonal,
    "Duplicate"                 : Duplicate,
    "Fold"                      : Fold,
    "Fork"                      : Fork,
    "FrameAsDict"               : FrameAsDict,
    "FrameAsTensor"             : FrameAsTensor,
    "FrameAsTensorDict"         : FrameAsTensorDict,
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
    "Pipe"                      : Pipe,
    "PositionalEncoder"         : PositionalEncoder,
    "Replicate"                 : Replicate,
    "StandardScaler"            : StandardScaler,
    "TensorConcatenator"        : TensorConcatenator,
    "TensorSplitter"            : TensorSplitter,
    "TimeDeltaEncoder"          : TimeDeltaEncoder,
    "TupleUnwrapper"            : TupleUnwrapper,
    "TupleWrapper"              : TupleWrapper,
    "WrappedEncoder"            : WrappedEncoder,
    **{f"pandas.{name}": cls for name, cls in pandas.ENCODERS.items()},
    **{f"polars.{name}": cls for name, cls in polars.ENCODERS.items()},
}  # fmt: skip
r"""Dictionary of all available encoders."""
