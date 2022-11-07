r"""Generic types for type hints etc."""

__all__ = [
    # Submodules
    "abc",
    "protocols",
    "time",
    # Type Variables
    "AliasVar",
    "AnyTypeVar",
    "ClassVar",
    "DtypeVar",
    "KeyVar",
    "Key_co",
    "ModuleVar",
    "NestedKeyVar",
    "NestedKey_co",
    "ObjectVar",
    "PandasVar",
    "PathVar",
    "ReturnVar",
    "Self",
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
    "Parameters",
]


from tsdm.utils.types import abc, protocols, time
from tsdm.utils.types._types import (
    AliasVar,
    AnyTypeVar,
    Args,
    ClassVar,
    DtypeVar,
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
    Parameters,
    PathType,
    PathVar,
    ReturnVar,
    Self,
    T,
    T_co,
    T_contra,
    TensorVar,
    TorchModuleVar,
    ValueVar,
)
