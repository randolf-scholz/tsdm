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
    "box_cox",
    "converters",
    "pandas",
    "polars",
    "positional",
    "torch",
    "universal",
]

from . import (
    base,
    box_cox,
    converters,
    pandas,
    polars,
    positional,
    torch,
    universal,
)
from .base import *  # ruff: ignore[F403]
from .box_cox import *  # ruff: ignore[F403]
from .converters import *  # ruff: ignore[F403]
from .positional import *  # ruff: ignore[F403]
from .universal import *  # ruff: ignore[F403]

__all__ += base.__all__
__all__ += box_cox.__all__
__all__ += converters.__all__
__all__ += positional.__all__
__all__ += universal.__all__

ENCODERS: dict[str, type[base.BaseEncoder]] = {
    # base
    "Choice"                    : base.Choice,
    "Compose"                   : base.Compose,
    "DeepcopyEncoder"           : base.DeepcopyEncoder,
    "Diagonal"                  : base.Diagonal,
    "Duplicate"                 : base.Duplicate,
    "Fold"                      : base.Fold,
    "Fork"                      : base.Fork,
    "IdentityEncoder"           : base.IdentityEncoder,
    "InverseEncoder"            : base.InverseEncoder,
    "MappedEncoder"             : base.MappedEncoder,
    "Meet"                      : base.Meet,
    "Parallel"                  : base.Parallel,
    "Pipe"                      : base.Pipe,
    "Replicate"                 : base.Replicate,
    "TupleUnwrapper"            : base.TupleUnwrapper,
    "TupleWrapper"              : base.TupleWrapper,
    "WrappedEncoder"            : base.WrappedEncoder,
    # box cox
    "BoxCoxEncoder"             : box_cox.BoxCoxEncoder,
    "LogEncoder"                : box_cox.LogEncoder,
    "LogitBoxCoxEncoder"        : box_cox.LogitBoxCoxEncoder,
    "LogitEncoder"              : box_cox.LogitEncoder,
    # converters
    "FrameDTypeConverter"       : converters.FrameDTypeConverter,
    "FrameAsDict"               : converters.FrameAsDict,
    "FrameAsTensor"             : converters.FrameAsTensor,
    "FrameAsTensorDict"         : converters.FrameAsTensorDict,
    # positional
    "PositionalEncoder"         : positional.PositionalEncoder,
    **universal.ENCODERS,
    **{f"pandas.{name}": cls for name, cls in pandas.ENCODERS.items()},
    **{f"polars.{name}": cls for name, cls in polars.ENCODERS.items()},
    **{f"torch.{name}": cls for name, cls in torch.ENCODERS.items()},
}  # fmt: skip
r"""Dictionary of all available encoders."""
