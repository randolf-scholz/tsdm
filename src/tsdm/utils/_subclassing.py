r"""Code for abstract attributes."""

__all__ = [
    "abstractattribute",
    "DummyAttribute",
    "PatchedABCMeta",
]

from abc import ABCMeta
from collections.abc import Callable

from typing_extensions import Any, Optional, TypeVar, cast

R = TypeVar("R")


class DummyAttribute:
    r"""Sentinel for abstract attributes."""

    __is_abstract_attribute__ = True


def abstractattribute(obj: Optional[Callable[[Any], R]] = None) -> R:
    r"""Decorate method as abstract attribute."""
    attr = DummyAttribute() if obj is None else obj
    try:
        attr.__is_abstract_attribute__ = True  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise AttributeError(
            f"Cannot decorate with abstractattribute decorator because {obj} "
            "does not support setting attributes."
        ) from exc
    return cast(R, attr)


# def attribute(obj: Callable[[T], R] = None) -> R:
#     r"""Decorator for attributes."""
#
#
# def classattribute(obj: Callable[[T], R] = None) -> R:
#     r"""Decorator equivalent of @attribute@classmethod."""
#
#
# def staticattribute(obj: Callable[[T], R] = None) -> R:
#     r"""Decorator equivalent of @attribute@staticmethod."""


class PatchedABCMeta(ABCMeta):
    r"""Patched ABCMeta class to allow @abstractattribute."""

    def __call__(cls, *args: Any, **kwargs: Any) -> "PatchedABCMeta":
        r"""Override __call__ to allow @abstractattribute."""
        instance: PatchedABCMeta = ABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), "__is_abstract_attribute__", False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                f"Can't instantiate abstract class {cls.__name__} with"
                f" abstract attributes: f{', '.join(abstract_attributes)}"
            )
        return instance
