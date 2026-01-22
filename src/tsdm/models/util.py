r"""Utilities for models."""

__all__ = ["autojit", "initialize_from_config"]

from functools import wraps
from importlib import import_module
from typing import Any, Self

from torch import jit, nn

from tsdm.config import CONFIG


def autojit[M: nn.Module](base_class: type[M], /) -> type[M]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule: ...


        model = jit.script(MyModule())

    and

    .. code-block:: python

        @autojit
        class MyModule: ...


        model = MyModule()

    are (roughly?) equivalent
    """
    if not isinstance(base_class, type):
        raise TypeError("Expected a class.")
    if not issubclass(base_class, nn.Module):
        raise TypeError("Expected a subclass of nn.Module.")

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type,misc]
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            # Note: If __new__() does not return an instance of cls,
            #   then the new instance's __init__() method will not be invoked.
            instance = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted = jit.script(instance)
                return scripted  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
            return instance  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    if not isinstance(WrappedClass, type):
        raise TypeError(f"Expected a class, got {WrappedClass}.")
    if not issubclass(WrappedClass, base_class):
        raise TypeError(f"Expected {WrappedClass} to be a subclass of {base_class}.")

    return WrappedClass  # pyright: ignore[reportReturnType]


def initialize_from_config(config: dict[str, Any], /) -> nn.Module:
    r"""Initialize `nn.Module` from a config object."""
    conf = config.copy()
    cls_name: str = conf.pop("__name__")
    module_name: str = conf.pop("__module__")

    # drop other dunder keys
    opts = {k: v for k, v in conf.items() if not k.startswith("__")}

    # import module and class
    module = import_module(module_name)
    cls = getattr(module, cls_name)

    # initialize class with options
    try:
        obj = cls(**opts)
    except Exception as exc:
        exc.add_note(f"Failed to initialize {cls_name} with {opts}.")
        raise

    return obj
