r"""Generic types for type hints, etc."""

__all__ = [
    # Custom Type Aliases
    "alias_var",
    "dtype_var",
    "pandas_var",
    "tensor_var",
    "torch_module_var",
    "scalar_var",
    "scalar_co",
    "scalar_contra",
    # Builtin Type Variables
    "class_var",
    "class_co",
    "class_contra",
    "dict_var",
    "dict_co",
    "dict_contra",
    "float_var",
    "float_co",
    "float_contra",
    "int_var",
    "int_co",
    "int_contra",
    "module_var",
    "module_co",
    "module_contra",
    "list_var",
    "list_co",
    "list_contra",
    "object_var",
    "object_co",
    "object_contra",
    "path_var",
    "path_co",
    "path_contra",
    "set_var",
    "set_co",
    "set_contra",
    "str_var",
    "str_co",
    "str_contra",
    "tuple_var",
    "tuple_co",
    "tuple_contra",
    # Generic Type Variables
    "P",
    "P2",
    "Cls",
    "Cls_co",
    "Cls_contra",
    "Fun",
    "Fun_co",
    "Fun_contra",
    "K",
    "K2",
    "K2_co",
    "K2_contra",
    "K_co",
    "K_contra",
    "R_co",
    "T",
    "T2",
    "T2_co",
    "T2_contra",
    "T_co",
    "T_contra",
    "V",
    "V_co",
    "V_contra",
    # collections.abc Type Variables
    "ByteStringType",
    "ByteString_co",
    "ByteString_contra",
    "CallableType",
    "Callable_co",
    "Callable_contra",
    "CollectionType",
    "Collection_co",
    "Collection_contra",
    "ContainerType",
    "Container_co",
    "Container_contra",
    "GeneratorType",
    "Generator_co",
    "Generator_contra",
    "HashableType",
    "Hashable_co",
    "Hashable_contra",
    "IterableType",
    "Iterable_co",
    "Iterable_contra",
    "IteratorType",
    "Iterator_co",
    "Iterator_contra",
    "MappingType",
    "Mapping_co",
    "Mapping_contra",
    "MutableMappingType",
    "MutableMapping_co",
    "MutableMapping_contra",
    "MutableSequenceType",
    "MutableSequence_co",
    "MutableSequence_contra",
    "MutableSetType",
    "MutableSet_co",
    "MutableSet_contra",
    "ReversibleType",
    "Reversible_co",
    "Reversible_contra",
    "SequenceType",
    "Sequence_co",
    "Sequence_contra",
    "SetType",
    "Set_co",
    "Set_contra",
    "SizedType",
    "Sized_co",
    "Sized_contra",
]

from collections import abc
from pathlib import Path
from types import GenericAlias, ModuleType

from numpy import ndarray
from pandas import DataFrame, Index, MultiIndex, Series
from torch import Tensor, nn
from typing_extensions import Any, ParamSpec, TypeVar

# region Custom Type Variables ---------------------------------------------------------
alias_var = TypeVar("alias_var", bound=GenericAlias)
r"""Type Variable `TypeAlias`."""
dtype_var = TypeVar("dtype_var")
r"""Type Variable for `Dtype`."""
pandas_var = TypeVar("pandas_var", Index, Series, DataFrame, MultiIndex)
r"""Type Variable for `pandas` objects."""
tensor_var = TypeVar("tensor_var", Tensor, ndarray)
r"""Type Variable for `torch.Tensor` or `numpy.ndarray` objects."""
torch_module_var = TypeVar("torch_module_var", bound=nn.Module)
r"""Type Variable for `torch.nn.Modules`."""
scalar_var = TypeVar("scalar_var")
r"""Type Variable for scalar types."""
scalar_co = TypeVar("scalar_co", covariant=True)
r"""Covariant type variable for scalar types."""
scalar_contra = TypeVar("scalar_contra", contravariant=True)
r"""Contravariant type variable for scalar types."""
# endregion Custom Type Variables ------------------------------------------------------

# Type variables of Built-in Types
# region Builtins ----------------------------------------------------------------------
class_var = TypeVar("class_var", bound=type)
r"""Type Variable for `type`."""
class_co = TypeVar("class_co", bound=type, covariant=True)
r"""Type Variable for `type`."""
class_contra = TypeVar("class_contra", bound=type, contravariant=True)
r"""Type Variable for `type`."""

dict_var = TypeVar("dict_var", bound=dict)
r"""Type Variable for `dict`."""
dict_co = TypeVar("dict_co", bound=dict, covariant=True)
r"""Covariant type variable for `dict`."""
dict_contra = TypeVar("dict_contra", bound=dict, contravariant=True)
r"""Contravariant type variable for `dict`."""

float_var = TypeVar("float_var", bound=float)
r"""Type Variable for `float`."""
float_co = TypeVar("float_co", bound=float, covariant=True)
r"""Covariant type variable for `float`."""
float_contra = TypeVar("float_contra", bound=float, contravariant=True)
r"""Contravariant type variable for `float`."""

int_var = TypeVar("int_var", bound=int)
r"""Type Variable for `int`."""
int_co = TypeVar("int_co", bound=int, covariant=True)
r"""Covariant type variable for `int`."""
int_contra = TypeVar("int_contra", bound=int, contravariant=True)
r"""Contravariant type variable for `int`."""

module_var = TypeVar("module_var", bound=ModuleType)
r"""Type Variable for `ModuleType`."""
module_co = TypeVar("module_co", bound=ModuleType, covariant=True)
r"""Covariant type variable for `ModuleType`."""
module_contra = TypeVar("module_contra", bound=ModuleType, contravariant=True)
r"""Contravariant type variable for `ModuleType`."""

list_var = TypeVar("list_var", bound=list)
r"""Type Variable for `list`."""
list_co = TypeVar("list_co", bound=list, covariant=True)
r"""Covariant type variable for `list`."""
list_contra = TypeVar("list_contra", bound=list, contravariant=True)
r"""Contravariant type variable for `list`."""

object_var = TypeVar("object_var", bound=object)
r"""Type Variable for `object`."""
object_co = TypeVar("object_co", bound=object, covariant=True)
r"""Covariant type variable for `object`."""
object_contra = TypeVar("object_contra", bound=object, contravariant=True)
r"""Contravariant type variable for `object`."""

path_var = TypeVar("path_var", bound=Path)
r"""Type Variable for `Path`."""
path_co = TypeVar("path_co", bound=Path, covariant=True)
r"""Covariant type variable for `Path`."""
path_contra = TypeVar("path_contra", bound=Path, contravariant=True)
r"""Contravariant type variable for `Path`."""

set_var = TypeVar("set_var", bound=set)
r"""Type Variable for `set`."""
set_co = TypeVar("set_co", bound=set, covariant=True)
r"""Covariant type variable for `set`."""
set_contra = TypeVar("set_contra", bound=set, contravariant=True)
r"""Contravariant type variable for `set`."""

str_var = TypeVar("str_var", bound=str)
r"""Type Variable for `str`."""
str_co = TypeVar("str_co", bound=str, covariant=True)
r"""Covariant type variable for `str`."""
str_contra = TypeVar("str_contra", bound=str, contravariant=True)
r"""Contravariant type variable for `str`."""

tuple_var = TypeVar("tuple_var", bound=tuple)
r"""Type Variable for `tuple`."""
tuple_co = TypeVar("tuple_co", bound=tuple, covariant=True)
r"""Covariant type variable for `tuple`."""
tuple_contra = TypeVar("tuple_contra", bound=tuple, contravariant=True)
r"""Contravariant type variable for `tuple`."""
# endregion Builtins -------------------------------------------------------------------


# Generic purpose type variables
# region Generic Type Variables --------------------------------------------------------
P = ParamSpec("P")
P2 = ParamSpec("P2")

T = TypeVar("T", bound=Any)
r"""T(ype) variable (invariant)."""
T_co = TypeVar("T_co", bound=Any, covariant=True)
r"""T(ype) variable (covariant)."""
T_contra = TypeVar("T_contra", bound=Any, contravariant=True)
r"""T(ype) variable (contravariant)."""

T2 = TypeVar("T2", bound=Any)
r"""T(ype) variable (invariant)."""
T2_co = TypeVar("T2_co", bound=Any, covariant=True)
r"""T(ype) variable (covariant)."""
T2_contra = TypeVar("T2_contra", bound=Any, contravariant=True)
r"""T(ype) variable (contravariant)."""

K = TypeVar("K", bound=abc.Hashable)
r"""K(ey) type variable bound to `Hashable` (invariant)."""
K_co = TypeVar("K_co", bound=abc.Hashable, covariant=True)
r"""K(ey) type variable bound to `Hashable` (covariant)."""
K_contra = TypeVar("K_contra", bound=abc.Hashable, contravariant=True)
r"""K(ey) type variable bound to `Hashable` (contravariant)."""

K2 = TypeVar("K2", bound=abc.Hashable)
r"""K(ey) type variable bound to `Hashable` (invariant)."""
K2_co = TypeVar("K2_co", bound=abc.Hashable, covariant=True)
r"""K(ey) type variable bound to `Hashable` (covariant)."""
K2_contra = TypeVar("K2_contra", bound=abc.Hashable, contravariant=True)
r"""K(ey) type variable bound to `Hashable` (contravariant)."""

Fun = TypeVar("Fun", bound=abc.Callable)
r"""Fun(ction) type variable (invariant)."""
Fun_co = TypeVar("Fun_co", bound=abc.Callable, covariant=True)
r"""Fun(ction) type variable (covariant)."""
Fun_contra = TypeVar("Fun_contra", bound=abc.Callable, contravariant=True)
r"""Fun(ction) type variable (contravariant)."""

Cls = TypeVar("Cls", bound=type)
r"""Cls type variable (invariant)."""
Cls_co = TypeVar("Cls_co", bound=type, covariant=True)
r"""Cls type variable (covariant)."""
Cls_contra = TypeVar("Cls_contra", bound=type, contravariant=True)
r"""Cls type variable (contravariant)."""

# outputs/returns are always covariant!
R_co = TypeVar("R_co", covariant=True)
r"""R(eturn) type variable (always covariant)."""

V = TypeVar("V")
r"""V(alue) type variable (invariant)."""
V_co = TypeVar("V_co", covariant=True)
r"""V(alue) type variable (covariant)."""
V_contra = TypeVar("V_contra", contravariant=True)
r"""V(alue) type variable (contravariant)."""
# endregion Generic Type Variables -----------------------------------------------------


# region collections.abc Type Variables  -----------------------------------------------
# region collections.abc Generic Type Variables  ---------------------------------------
ContainerType = TypeVar("ContainerType", bound=abc.Container)
r"""Type variable for Containers."""
Container_co = TypeVar("Container_co", bound=abc.Container, covariant=True)
r"""Type variable for Containers."""
Container_contra = TypeVar("Container_contra", bound=abc.Container, contravariant=True)
r"""Type variable for Containers."""


HashableType = TypeVar("HashableType", bound=abc.Hashable)
r"""Type variable for Hashable objects."""
Hashable_co = TypeVar("Hashable_co", bound=abc.Hashable, covariant=True)
r"""Type variable for Hashable objects."""
Hashable_contra = TypeVar("Hashable_contra", bound=abc.Hashable, contravariant=True)
r"""Type variable for Hashable objects."""


IterableType = TypeVar("IterableType", bound=abc.Iterable)
r"""Type variable for Iterables."""
Iterable_co = TypeVar("Iterable_co", bound=abc.Iterable, covariant=True)
r"""Type variable for Iterables."""
Iterable_contra = TypeVar("Iterable_contra", bound=abc.Iterable, contravariant=True)
r"""Type variable for Iterables."""


IteratorType = TypeVar("IteratorType", bound=abc.Iterator)
r"""Type variable for Iterators."""
Iterator_co = TypeVar("Iterator_co", bound=abc.Iterator, covariant=True)
r"""Type variable for Iterators."""
Iterator_contra = TypeVar("Iterator_contra", bound=abc.Iterator, contravariant=True)
r"""Type variable for Iterators."""


ReversibleType = TypeVar("ReversibleType", bound=abc.Reversible)
r"""Type variable for Reversible."""
Reversible_co = TypeVar("Reversible_co", bound=abc.Reversible, covariant=True)
r"""Type variable for Reversible."""
Reversible_contra = TypeVar(
    "Reversible_contra", bound=abc.Reversible, contravariant=True
)
r"""Type variable for Reversible."""


GeneratorType = TypeVar("GeneratorType", bound=abc.Generator)
r"""Type variable for Generators."""
Generator_co = TypeVar("Generator_co", bound=abc.Generator, covariant=True)
r"""Type variable for Generators."""
Generator_contra = TypeVar("Generator_contra", bound=abc.Generator, contravariant=True)
r"""Type variable for Generators."""


SizedType = TypeVar("SizedType", bound=abc.Sized)
r"""Type variable for Mappings."""
Sized_co = TypeVar("Sized_co", bound=abc.Sized, covariant=True)
r"""Type variable for Mappings."""
Sized_contra = TypeVar("Sized_contra", bound=abc.Sized, contravariant=True)
r"""Type variable for Mappings."""


CallableType = TypeVar("CallableType", bound=abc.Callable)
r"""Type variable for Callables."""
Callable_co = TypeVar("Callable_co", bound=abc.Callable, covariant=True)
r"""Type variable for Callables."""
Callable_contra = TypeVar("Callable_contra", bound=abc.Callable, contravariant=True)
r"""Type variable for Callables."""


CollectionType = TypeVar("CollectionType", bound=abc.Collection)
r"""Type variable for Collections."""
Collection_co = TypeVar("Collection_co", bound=abc.Collection, covariant=True)
r"""Type variable for Collections."""
Collection_contra = TypeVar(
    "Collection_contra", bound=abc.Collection, contravariant=True
)
r"""Type variable for Collections."""


SequenceType = TypeVar("SequenceType", bound=abc.Sequence)
r"""Type variable for Sequences."""
Sequence_co = TypeVar("Sequence_co", bound=abc.Sequence, covariant=True)
r"""Type variable for Sequences."""
Sequence_contra = TypeVar("Sequence_contra", bound=abc.Sequence, contravariant=True)
r"""Type variable for Sequences."""


MutableSequenceType = TypeVar("MutableSequenceType", bound=abc.MutableSequence)
r"""Type variable for MutableSequences."""
MutableSequence_co = TypeVar(
    "MutableSequence_co", bound=abc.MutableSequence, covariant=True
)
r"""Type variable for MutableSequences."""
MutableSequence_contra = TypeVar(
    "MutableSequence_contra", bound=abc.MutableSequence, contravariant=True
)
r"""Type variable for MutableSequences."""


ByteStringType = TypeVar("ByteStringType", bound=abc.ByteString)
r"""Type variable for ByteStrings."""
ByteString_co = TypeVar("ByteString_co", bound=abc.ByteString, covariant=True)
r"""Type variable for ByteStrings."""
ByteString_contra = TypeVar(
    "ByteString_contra", bound=abc.ByteString, contravariant=True
)
r"""Type variable for ByteStrings."""


SetType = TypeVar("SetType", bound=abc.Set)
r"""Type variable for Sets."""
Set_co = TypeVar("Set_co", bound=abc.Set, covariant=True)
r"""Type variable for Sets."""
Set_contra = TypeVar("Set_contra", bound=abc.Set, contravariant=True)
r"""Type variable for Sets."""


MutableSetType = TypeVar("MutableSetType", bound=abc.MutableSet)
r"""Type variable for MutableSets."""
MutableSet_co = TypeVar("MutableSet_co", bound=abc.MutableSet, covariant=True)
r"""Type variable for MutableSets."""
MutableSet_contra = TypeVar(
    "MutableSet_contra", bound=abc.MutableSet, contravariant=True
)
r"""Type variable for MutableSets."""


MappingType = TypeVar("MappingType", bound=abc.Mapping)
r"""Type variable for Mappings."""
Mapping_co = TypeVar("Mapping_co", bound=abc.Mapping, covariant=True)
r"""Type variable for Mappings."""
Mapping_contra = TypeVar("Mapping_contra", bound=abc.Mapping, contravariant=True)
r"""Type variable for Mappings."""


MutableMappingType = TypeVar("MutableMappingType", bound=abc.MutableMapping)
r"""Type variable for MutableMappings."""
MutableMapping_co = TypeVar(
    "MutableMapping_co", bound=abc.MutableMapping, covariant=True
)
r"""Type variable for MutableMappings."""
MutableMapping_contra = TypeVar(
    "MutableMapping_contra", bound=abc.MutableMapping, contravariant=True
)
r"""Type variable for MutableMappings."""

# endregion collections.abc Generic Type Variables  ------------------------------------


# region collections.abc View Type Variables  ------------------------------------------
mapping_view_var = TypeVar("mapping_view_var", bound=abc.MappingView)
r"""Type variable for MappingViews."""
ItemsViewType = TypeVar("ItemsViewType", bound=abc.ItemsView)
r"""Type variable for ItemsViews."""
KeysViewType = TypeVar("KeysViewType", bound=abc.KeysView)
r"""Type variable for KeysViews."""
ValuesViewType = TypeVar("ValuesViewType", bound=abc.ValuesView)
r"""Type variable for ValuesViews."""
# endregion collections.abc View Type Variables  ---------------------------------------


# region collections.abc Async Type Variables  -----------------------------------------
AwaitableType = TypeVar("AwaitableType", bound=abc.Awaitable)
r"""Type variable for Awaitables."""
CoroutineType = TypeVar("CoroutineType", bound=abc.Coroutine)
r"""Type variable for Coroutines."""
AsyncIterableType = TypeVar("AsyncIterableType", bound=abc.AsyncIterable)
r"""Type variable for AsyncIterables."""
AsyncIteratorType = TypeVar("AsyncIteratorType", bound=abc.AsyncIterator)
r"""Type variable for AsyncIterators."""
AsyncGeneratorType = TypeVar("AsyncGeneratorType", bound=abc.AsyncGenerator)
r"""Type variable for AsyncGenerators."""
# endregion collections.abc Async Type Variables  --------------------------------------
# endregion collections.abc Type Variables  --------------------------------------------

del abc, TypeVar, ModuleType, GenericAlias, Path
del DataFrame, Index, MultiIndex, Series
del Tensor, nn, ndarray
