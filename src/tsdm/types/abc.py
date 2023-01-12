r"""Type Variables from collections.abc objects."""

__all__ = [
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
from typing import TypeVar

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
MappingViewType = TypeVar("MappingViewType", bound=abc.MappingView)
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

del abc, TypeVar
