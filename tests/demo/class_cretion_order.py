r"""Demonstrate class creation order."""
# mypy: ignore-errors


def class_decorator[T](cls: type[T], /) -> type[T]:
    r"""Create a decorator that converts class to decorator."""
    print(f"class_decorator({cls=})")
    return cls


class Meta(type):
    r"""Metaclass for class decorators."""

    def __call__(cls, /, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Meta.__call__({cls=})")
        return super().__call__(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Meta.__new__({cls=})")
        return super().__new__(cls, *args, **kwargs)

    def __init__(cls, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Meta.__init__({cls=})")
        super().__init__(*args, **kwargs)


print("------- Base Defintion -----------")


@class_decorator
class Base(metaclass=Meta):
    r"""Base class for class decorators."""

    def __new__(cls, /, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Base.__new__({cls=})")
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        r"""Create a decorator that converts class to decorator."""
        print(f"Base.__init__({self=})")
        super().__init__()

    def __init_subclass__(cls, /, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Base.__init_subclass__({cls=})")
        super().__init_subclass__(**kwargs)


print("------- Base Instantiation -----------")

base = Base()

print("------- Subclass Defintion -----------")


@class_decorator
class Subclass(Base):
    r"""Subclass for class decorators."""

    def __new__(cls, /, *args, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Subclass.__new__({cls=})")
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        r"""Create a decorator that converts class to decorator."""
        print(f"Subclass.__init__({self=})")
        super().__init__()

    def __init_subclass__(cls, /, **kwargs):
        r"""Create a decorator that converts class to decorator."""
        print(f"Subclass.__init_subclass__({cls=})")


print("------- Subclass Instantiation -----------")

sub = Subclass()
