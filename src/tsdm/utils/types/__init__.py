r"""Generic types for type hints etc."""

__all__ = [
    # Submodules
    "abc",
    "dtypes",
    "protocols",
    "time",
    # Type Variables
    "AliasVar",
    "AnyTypeVar",
    "ClassVar",
    "DtypeVar",
    "IDVar",
    "KeyVar",
    "Key_co",
    "ModuleVar",
    "NestedKeyVar",
    "NestedKey_co",
    "ObjectVar",
    "PandasVar",
    "PathVar",
    "ReturnVar",
    "T",
    "T_co",
    "T_contra",
    "TensorVar",
    "TorchModuleVar",
    "ValueVar",
    # Type Aliases
    "Args",
    "KWArgs",
    "Nested",
    "PandasObject",
    "PathType",
    # ParamSpec
    "ParameterVar",
]


from tsdm.utils.types import abc, dtypes, protocols, time
from tsdm.utils.types._types import (
    AliasVar,
    AnyTypeVar,
    Args,
    ClassVar,
    DtypeVar,
    IDVar,
    Key_co,
    KeyVar,
    KWArgs,
    ModuleVar,
    Nested,
    NestedKey_co,
    NestedKeyVar,
    ObjectVar,
    PandasObject,
    PandasVar,
    ParameterVar,
    PathType,
    PathVar,
    ReturnVar,
    T,
    T_co,
    T_contra,
    TensorVar,
    TorchModuleVar,
    ValueVar,
)
