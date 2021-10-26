r"""Utility functions.

TODO:  Module description
"""

from __future__ import annotations

__all__ = [
    # Constants
    # Classes
    "Split",
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "initialize_from",
    "now",
    "skew_symmetric",
    "relsize_skewpart",
    "symmetric",
    "relsize_symmpart",
]


from collections.abc import Mapping
from datetime import datetime
from functools import partial
from logging import getLogger
from typing import Any, Callable, Iterable, NamedTuple, Union, overload

import torch
from numpy.typing import NDArray
from torch import Tensor, jit

from tsdm.util.types import LookupTable, ObjectType, ReturnType

LOGGER = getLogger(__name__)


class Split(NamedTuple):
    r"""Holds indices for train/valid/test set."""

    train: NDArray
    valid: NDArray
    test: NDArray


def now():
    r"""Return current time in iso format."""
    return datetime.now().isoformat(timespec="seconds")


def deep_dict_update(d: dict, new_kvals: Mapping) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: Mapping
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in new_kvals.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


def deep_kval_update(d: dict, **new_kvals: dict) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: dict
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_kval_update(d.get(key, {}), **new_kvals)
        elif key in new_kvals:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


def flatten_dict(
    d: dict[Any, Iterable[Any]], recursive: bool = True
) -> list[tuple[Any, ...]]:
    r"""Flatten a dictionary containing iterables to a list of tuples.

    Parameters
    ----------
    d: dict
    recursive: bool (default=True)
        If true applies flattening strategy recursively on nested dicts, yielding
        list[tuple[key1, key2, ...., keyN, value]]

    Returns
    -------
    list[tuple[Any, ...]]
    """
    result = []
    for key, iterable in d.items():
        for item in iterable:
            if isinstance(item, dict) and recursive:
                gen: list[tuple[Any, ...]] = flatten_dict(item, recursive=True)
                result += [(key,) + tup for tup in gen]
            else:
                result += [(key, item)]
    return result


# T = TypeVar("T")
# S = TypeVar("S")


# ModularTable = dict[str, type[T]]
# FunctionalTable = dict[str, Callable[..., S]]
# LookupTable = Union[
#     ModularTable, FunctionalTable, dict[str, Union[type[T], Callable[..., S]]]
# ]


# @overload
# def initialize_from(
#     lookup_table: LookupTable[ObjectType], __name__: str, **kwargs: Any
# ) -> ObjectType:
#     ...


# partial from func
@overload
def initialize_from(
    lookup_table: LookupTable[Callable[..., ReturnType]],
    /,
    __name__: str,
    **kwargs: Any,
) -> Callable[..., ReturnType]:
    ...


# partial from type
@overload
def initialize_from(
    lookup_table: LookupTable[type[ObjectType]],
    /,
    __name__: str,
    **kwargs: Any,
) -> ObjectType:
    ...


# partial from already initialized object
@overload
def initialize_from(
    lookup_table: LookupTable[ObjectType],
    /,
    __name__: str,
    **kwargs: Any,
) -> ObjectType:
    ...


@overload
def initialize_from(
    lookup_table: LookupTable[Union[Callable[..., ReturnType], type[ObjectType]]],
    /,
    __name__: str,
    **kwargs: Any,
) -> Union[Callable[..., ReturnType], ObjectType]:
    ...


# @overload
# def initialize_from(
#     lookup_table: LookupTable[Union[Callable[..., ReturnType], ObjectType]],
#     /,
#     __name__: str,
#     **kwargs: Any,
# ) -> Union[Callable[..., ReturnType], ObjectType]:
#     ...


# @overload
# def initialize_from(
#     lookup_table: LookupTable[ObjectType],
#     /,
#     __name__: str,
#     **kwargs: Any,
# ) -> ObjectType:
#     ...


def initialize_from(
    lookup_table,
    /,
    __name__: str,
    **kwargs: Any,
):
    r"""Lookup class/function from dictionary and initialize it.

    Roughly equivalent to:

    .. code-block:: python

        obj = lookup_table[__name__]
        if isclass(obj):
            return obj(**kwargs)
        return partial(obj, **kwargs)

    Parameters
    ----------
    lookup_table: dict[str, Callable]
    __name__: str
        The name of the class/function
    kwargs: Any
        Optional arguments to initialize class/function

    Returns
    -------
    Callable
        The initialized class/function
    """
    obj = lookup_table[__name__]
    assert callable(obj), f"Looked up object {obj} not callable class/function."

    # check that obj is a class, but not metaclass or instance.
    if isinstance(obj, type) and not issubclass(obj, type):
        return obj(**kwargs)  # type: ignore[call-arg]
    # if it is function, fix kwargs
    return partial(obj, **kwargs)


# @initialize_from.register  # type: ignore[no-redef]
# # Modular Only
# def _(lookup_table: dict[str, type[T]], __name__: str, **kwargs: Any) -> T:
#     obj = lookup_table[__name__]
#     assert callable(obj), f"Looked up object {obj} not callable class/function."
#     return obj(**kwargs)
#
#
# @initialize_from.register  # type: ignore[no-redef]
# # Functional Only
# def _(lookup_table: dict[str, Callable[..., S]], __name__: str, **kwargs: Any) -> S:
#     obj = lookup_table[__name__]
#     assert callable(obj), f"Looked up object {obj} not callable class/function."
#     return partial(obj, **kwargs)
#
#
# @initialize_from.register  # type: ignore[no-redef]
# # Both Modular or Functional
# def _(
#     lookup_table: dict[str, Union[type[T], Callable[..., S]]],
#     __name__: str,
#     **kwargs: Any,
# ) -> Union[T, Callable[..., S]]:
#     obj = lookup_table[__name__]
#     assert callable(obj), f"Looked up object {obj} not callable class/function."
#
#     # check that obj is a class, but not metaclass or instance.
#     if isinstance(obj, type) and not issubclass(obj, type):
#         return obj(**kwargs)
#     return partial(obj, **kwargs)
#
#

# @singledispatch
# def initialize_from(lookup_table, __name__: str, **kwargs: Any):
#     """Lookup class/function from dictionary and initialize it.
#
#     Roughly equivalent to:
#
#     .. code-block:: python
#
#         obj = lookup_table[__name__]
#         if isclass(obj):
#             return obj(**kwargs)
#         return partial(obj, **kwargs)
#
#
#     Parameters
#     ----------
#     lookup_table: dict[str, Callable]
#     __name__: str
#         The name of the class/function
#     kwargs: Any
#         Optional arguments to initialize class/function
#
#     Returns
#     -------
#     Callable
#         The initialized class/function
#     """
#     ...
#
#
# @initialize_from.register  # type: ignore[no-redef]
# # Modular Only
# def _(lookup_table: dict[str, type[T]], __name__: str, **kwargs: Any) -> T:
#     obj = lookup_table[__name__]
#     assert callable(obj), f"Looked up object {obj} not callable class/function."
#     return obj(**kwargs)
#
#
# @initialize_from.register  # type: ignore[no-redef]
# # Functional Only
# def _(lookup_table: dict[str, Callable[..., S]], __name__: str, **kwargs: Any) -> S:
#     obj = lookup_table[__name__]
#     assert callable(obj), f"Looked up object {obj} not callable class/function."
#     return partial(obj, **kwargs)
#
#
# @initialize_from.register  # type: ignore[no-redef]
# # Both Modular or Functional
# def _(
#     lookup_table: dict[str, Union[type[T], Callable[..., S]]],
#     __name__: str,
#     **kwargs: Any,
# ) -> Union[T, Callable[..., S]]:
#     obj = lookup_table[__name__]
#     assert callable(obj), f"Looked up object {obj} not callable class/function."
#
#     # check that obj is a class, but not metaclass or instance.
#     if isinstance(obj, type) and not issubclass(obj, type):
#         return obj(**kwargs)
#     return partial(obj, **kwargs)
@jit.script
def symmetric(x: Tensor) -> Tensor:
    r"""Symmetric part of matrix."""
    return (x + x.swapaxes(-1, -2)) / 2


@jit.script
def skew_symmetric(x: Tensor) -> Tensor:
    r"""Skew-Symmetric part of matrix."""
    return (x - x.swapaxes(-1, -2)) / 2


@jit.script
def relsize_symmpart(kernel):
    r"""Relative magnitude of symmetric part."""
    return torch.mean(symmetric(kernel) ** 2) / torch.mean(kernel ** 2)


@jit.script
def relsize_skewpart(kernel):
    r"""Relative magnitude of skew-symmetric part."""
    return torch.mean(skew_symmetric(kernel) ** 2) / torch.mean(kernel ** 2)
