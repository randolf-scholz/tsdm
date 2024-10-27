r"""Demonstrate class creation order."""
# mypy: ignore-errors
# pyright: basic


def class_decorator[T](cls: type[T], /) -> type[T]:
    r"""Create a decorator that converts class to decorator."""
    print(f"class_decorator({cls=})")
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


print("\n\n------- Base Defintion -----------")


@class_decorator
class Base(metaclass=Meta):
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


print("\n\n------- Base Instantiation -----------")

base = Base(1, 2, 3, foo="foo", bar="bar")

print("\n\n------- Subclass Defintion -----------")


@class_decorator
class Subclass(Base):
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


print("\n\n------- Subclass Instantiation -----------")

sub = Subclass(1, 2, 3, foo="foo", bar="bar")
