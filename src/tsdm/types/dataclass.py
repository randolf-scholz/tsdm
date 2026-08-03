r"""Dataclass protocol and utilities."""

__all__ = [
    "Dataclass",
    "is_dataclass",
    "isinstance_dataclass",
    "issubclass_dataclass",
    # type qualifiers
    "Fittable",
    "Derivable",
    "DerivedField",
    "FittedField",
]

import dataclasses
from typing import (
    Annotated,
    Any,
    ClassVar,
    Protocol,
    TypeIs,
    _ProtocolMeta as ProtocolMeta,
    overload,
    runtime_checkable,
)

# region type qualifiers ---------------------------------------------------------------
type Fittable[T] = Annotated[T, "Fittable"]
r"""Type Alias for fields that can be fitted."""
type Derivable[T] = Annotated[T, "Derivable"]
r"""Type Alias for fields that can be derived automatically."""
type DerivedField[T] = Annotated[T, "DerivedField"]
r"""Type Alias for fields that are derived automatically."""
type FittedField[T] = Annotated[T, "FittedField"]
r"""Type Alias for fields that are fitted automatically."""
# endregion type qualifiers ------------------------------------------------------------


class _DataclassMeta(ProtocolMeta):
    r"""Metaclass for `Dataclass`."""

    def __instancecheck__(cls, instance: object, /) -> TypeIs[Dataclass]:  # noqa: N805
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type, /) -> TypeIs[type[Dataclass]]:  # noqa: N805
        fields = getattr(subclass, "__dataclass_fields__", None)
        return isinstance(fields, dict)


@runtime_checkable
class Dataclass(Protocol, metaclass=_DataclassMeta):
    r"""Protocol for anonymous dataclasses.

    Similar to `DataClassInstance` from typeshed, but allows isinstance and issubclass.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]] = {}
    r"""The fields of the dataclass."""


def issubclass_dataclass(cls: type, /) -> TypeIs[type[Dataclass]]:
    return issubclass(cls, Dataclass)  # type: ignore[misc]


def isinstance_dataclass(obj: object, /) -> TypeIs[Dataclass]:
    return issubclass(type(obj), Dataclass)  # type: ignore[misc]


@overload
def is_dataclass(obj: type, /) -> TypeIs[type[Dataclass]]: ...
@overload
def is_dataclass(obj: object, /) -> TypeIs[Dataclass]: ...
def is_dataclass(obj: object, /) -> TypeIs[Dataclass] | TypeIs[type[Dataclass]]:
    r"""Check if the object is a dataclass."""
    if isinstance(obj, type):
        return issubclass(obj, Dataclass)  # type: ignore[misc]
    return issubclass(type(obj), Dataclass)  # type: ignore[misc]
