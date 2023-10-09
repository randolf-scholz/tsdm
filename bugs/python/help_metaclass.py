#!/usr/bin/env ipython


class ClassWithPrePostInit(type):
    def __pre_init__(cls) -> None:
        """Pre-initialization hook."""
        ...

    def __post_init__(cls) -> None:
        """Post-initialization hook."""
        ...

    def __call__(cls, *init_args, **init_kwargs):
        obj = cls.__new__(cls, *init_args, **init_kwargs)
        cls.__pre_init__(obj, *init_args, **init_kwargs)
        cls.__init__(obj, *init_args, **init_kwargs)
        cls.__post_init__(obj, *init_args, **init_kwargs)
        return obj


class SomeClass(metaclass=ClassWithPrePostInit):
    def __pre_init__(self, x: float) -> None:
        assert x > 0

    def __init__(self, x: float) -> None:
        self.x = x**2

    def __post_init__(self, x: float) -> None:
        assert self.x > 0


help(SomeClass)
