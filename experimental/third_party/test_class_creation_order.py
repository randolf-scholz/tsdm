r"""Demonstrate class creation order."""
# mypy: ignore-errors
# pyright: basic

from dataclasses import dataclass, is_dataclass


def class_decorator[T](cls: type[T], /) -> type[T]:
    r"""Create a decorator that converts class to decorator."""
    print(f"class_decorator(\n\t{cls=})\n")
    return cls


class Meta(type):
    r"""Metaclass for class decorators."""

    def __call__(cls, /, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Meta.__call__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
        return super().__call__(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Meta.__new__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
        return super().__new__(cls, *args, **kwargs)

    def __init__(cls, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Meta.__init__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
        super().__init__(*args, **kwargs)


def test_dataclass_subclass_definition() -> None:
    print("\n\nClass definition...\n")

    @class_decorator
    class Base(metaclass=Meta):
        r"""Base class for class decorators."""

        def __new__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__new__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            print(f"{is_dataclass(cls)=}")
            return super().__new__(cls)

        def __init__(self, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init__(\n\t{self=}\n\t{args=}\n\t{kwargs=})")
            super().__init__()

        def __init_subclass__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init_subclass__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            print(f"{is_dataclass(cls)=}")
            super().__init_subclass__(**kwargs)

    @dataclass
    class ExampleSubclass(Base):
        r"""Subclass for class decorators."""

        def __new__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Subclass.__new__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            return super().__new__(cls)

        def __init__(self, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Subclass.__init__(\n\t{self=}\n\t{args=}\n\t{kwargs=})")
            super().__init__()

        def __init_subclass__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Subclass.__init_subclass__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")


def test_instantiate_subclass() -> None:
    r"""Test instantiation of the subclass."""

    @class_decorator
    class Base(metaclass=Meta):
        r"""Base class for class decorators."""

        def __new__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__new__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            print(f"{is_dataclass(cls)=}")
            return super().__new__(cls)

        def __init__(self, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init__(\n\t{self=}\n\t{args=}\n\t{kwargs=})")
            super().__init__()

        def __init_subclass__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init_subclass__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            print(f"{is_dataclass(cls)=}")
            super().__init_subclass__(**kwargs)

    print("\n\nSubclass instantiation...\n")

    @class_decorator
    class Subclass(Base):
        r"""Subclass for class decorators."""

        def __new__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Subclass.__new__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            print(f"{is_dataclass(cls)=}")
            return super().__new__(cls)

        def __init__(self, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Subclass.__init__(\n\t{self=}\n\t{args=}\n\t{kwargs=})")
            super().__init__()

        def __init_subclass__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Subclass.__init_subclass__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            print(f"{is_dataclass(cls)=}")
            super().__init_subclass__(**kwargs)

    print("\n\nInstantiating Subclass...\n")
    Subclass(1, 2, 3, foo="foo", bar="bar")


def test_class_definition() -> None:
    print("\n\nClass definition...\n")

    @class_decorator
    class ExampleClass(metaclass=Meta):
        r"""Base class for class decorators."""

        def __new__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__new__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            return super().__new__(cls)

        def __init__(self, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init__(\n\t{self=}\n\t{args=}\n\t{kwargs=})")
            super().__init__()

        def __init_subclass__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init_subclass__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            super().__init_subclass__(**kwargs)


def test_dataclass_definition() -> None:
    print("\n\nDataclass definition...\n")

    @dataclass
    class ExampleClass(metaclass=Meta):
        r"""Base class for class decorators."""

        def __new__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__new__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            return super().__new__(cls)

        def __init__(self, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init__(\n\t{self=}\n\t{args=}\n\t{kwargs=})")
            super().__init__()

        def __init_subclass__(cls, /, *args, **kwargs):
            r"""Create a decorator that converts class to decorator."""
            print(f"Base.__init_subclass__(\n\t{cls=}\n\t{args=}\n\t{kwargs=})")
            super().__init_subclass__(**kwargs)
