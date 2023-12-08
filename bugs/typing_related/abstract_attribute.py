from abc import abstractmethod
from functools import cached_property
from typing import ClassVar, Final, Protocol


class ParentAttribute(Protocol):
    foo: int


class ParentFinalAttribute(Protocol):
    foo: Final[int]


class ParentClassVar(Protocol):
    foo: ClassVar[int]


class ParentProperty(Protocol):
    @property
    @abstractmethod
    def foo(self) -> int: ...


class ParentCachedProperty(Protocol):
    @cached_property
    @abstractmethod
    def foo(self) -> int: ...


class Attribute_Attribute(ParentAttribute):
    foo: int = 0


class Attribute_FinalAttribute(ParentFinalAttribute):
    foo: int = 0


class Attribute_ClassVar(ParentAttribute):
    foo: ClassVar[int] = 0


class Attribute_Property(ParentAttribute):
    @property
    def foo(self) -> int:
        return 0


class Attribute_CachedProperty(ParentAttribute):
    @cached_property
    def foo(self) -> int:
        return 0


class ClassVar_Attribute(ParentClassVar):
    foo: int = 0


class ClassVar_FinalAttribute(ParentClassVar):
    foo: Final[int] = 0


class ClassVar_ClassVar(ParentClassVar):
    foo: ClassVar[int] = 0


class ClassVar_Property(ParentClassVar):
    @property
    def foo(self) -> int:
        return 0


class ClassVar_CachedProperty(ParentClassVar):
    @cached_property
    def foo(self) -> int:
        return 0


class Property_Attribute(ParentProperty):
    foo: int = 0


class Property_FinalAttribute(ParentProperty):
    foo: Final[int] = 0


class Property_ClassVar(ParentProperty):
    foo: ClassVar[int] = 0


class Property_Property(ParentProperty):
    @property
    def foo(self) -> int:
        return 0


class Property_CachedProperty(ParentProperty):
    @cached_property
    def foo(self) -> int:
        return 0


class CachedProperty_Attribute(ParentCachedProperty):
    foo: int = 0


class CachedProperty_FinalAttribute(ParentCachedProperty):
    foo: Final[int] = 0


class CachedProperty_ClassVar(ParentCachedProperty):
    foo: ClassVar[int] = 0


class CachedProperty_Property(ParentCachedProperty):
    @property
    def foo(self) -> int:
        return 0


class CachedProperty_CachedProperty(ParentCachedProperty):
    @cached_property
    def foo(self) -> int:
        return 0
