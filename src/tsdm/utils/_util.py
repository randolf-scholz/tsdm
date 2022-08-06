r"""Utility functions.

TODO:  Module description
"""

__all__ = [
    # Constants
    # Classes
    "Split",
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "initialize_from",
    "initialize_from_config",
    "is_partition",
    "now",
    "paths_exists",
    "prepend_path",
    "round_relative",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
]

import os
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import datetime
from functools import partial
from importlib import import_module
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

from tsdm.utils.types import Nested, ObjectType, PathType, ReturnType
from tsdm.utils.types.abc import HashableType

__logger__ = getLogger(__name__)
EmptyPath: Path = Path()


def pairwise_disjoint(sets: Iterable[set]) -> bool:
    r"""Check if sets are pairwise disjoint."""
    union = set().union(*sets)
    return len(union) == sum(len(s) for s in sets)


def pairwise_disjoint_masks(masks: Iterable[NDArray[np.bool_]]) -> bool:
    r"""Check if masks are pairwise disjoint."""
    return all(sum(masks) == 1)  # type: ignore[arg-type]


def flatten_dict(d: Mapping) -> dict:
    r"""Flatten nested dictionary.

    Parameters
    ----------
    d: Mapping

    Returns
    -------
    dict
    """
    return {k: v for k, v in d.items() if not isinstance(v, Mapping)}


class Split(NamedTuple):
    r"""Holds indices for train/valid/test set."""

    train: Any
    valid: Any
    test: Any


def round_relative(x: np.ndarray, decimals: int = 2) -> np.ndarray:
    r"""Round to relative precision."""
    order = np.where(x == 0, 0, np.floor(np.log10(x)))
    digits = decimals - order
    rounded = np.rint(x * 10**digits)
    return np.true_divide(rounded, 10**digits)


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


def apply_nested(
    nested: Nested[Optional[ObjectType]],
    kind: type[ObjectType],
    func: Callable[[ObjectType], ReturnType],
) -> Nested[Optional[ReturnType]]:
    r"""Apply function to nested iterables of a given kind.

    Parameters
    ----------
    nested: Nested Data-Structure (Iterable, Mapping, ...)
    kind: The type of the leave nodes
    func: A function to apply to all leave Nodes
    """
    if nested is None:
        return None
    if isinstance(nested, kind):
        return func(nested)
    if isinstance(nested, Mapping):
        return {k: apply_nested(v, kind, func) for k, v in nested.items()}
    # TODO https://github.com/python/mypy/issues/11615
    if isinstance(nested, Collection):
        return [apply_nested(obj, kind, func) for obj in nested]  # type: ignore[misc]
    raise TypeError(f"Unsupported type: {type(nested)}")


@overload
def prepend_path(
    files: Nested[PathType],
    parent: Path,
    *,
    keep_none: bool = False,
) -> Nested[Path]:
    ...


@overload
def prepend_path(
    files: Nested[Optional[PathType]],
    parent: Path,
    *,
    keep_none: Literal[False] = False,
) -> Nested[Path]:
    ...


@overload
def prepend_path(
    files: Nested[Optional[PathType]],
    parent: Path,
    *,
    keep_none: Literal[True] = True,
) -> Nested[Optional[Path]]:
    ...


def prepend_path(
    files: Nested[Optional[PathType]],
    parent: Path,
    *,
    keep_none: bool = False,
) -> Nested[Optional[Path]]:
    r"""Prepends path to all files in nested iterable.

    Parameters
    ----------
    files
        Nested datastructes with Path-objects at leave nodes.
    parent: Path
    keep_none: bool
        If True, None-values are kept.

    Returns
    -------
    Nested data-structure with prepended path.
    """
    # TODO: change it to apply_nested in python 3.10

    if files is None:
        return None if keep_none else parent
    if isinstance(files, (str, Path, os.PathLike)):
        return parent / Path(files)
    if isinstance(files, Mapping):
        return {
            k: prepend_path(v, parent, keep_none=keep_none) for k, v in files.items()  # type: ignore[arg-type]
        }
    # TODO https://github.com/python/mypy/issues/11615
    if isinstance(files, Collection):
        return [prepend_path(f, parent, keep_none=keep_none) for f in files]  # type: ignore[misc,arg-type]
    raise TypeError(f"Unsupported type: {type(files)}")


def flatten_nested(nested: Any, kind: type[HashableType]) -> set[HashableType]:
    r"""Flatten nested iterables of a given kind.

    Parameters
    ----------
    nested: Any
    kind: hashable

    Returns
    -------
    set[hashable]
    """
    if nested is None:
        return set()
    if isinstance(nested, kind):
        return {nested}
    if isinstance(nested, Mapping):
        return set.union(*(flatten_nested(v, kind) for v in nested.values()))
    if isinstance(nested, Iterable):
        return set.union(*(flatten_nested(v, kind) for v in nested))
    raise ValueError(f"{type(nested)} is not understood")


# T = TypeVar("T")   \t\ \̃   \hy ŷ yy{̂y}    \tg g̃
# S = TypeVar("S")


# ModularTable = dict[str, type[T]]
# FunctionalTable = dict[str, Callable[..., S]]
# LookupTable = Union[
#     ModularTable, FunctionalTable, dict[str, Union[type[T], Callable[..., S]]]
# ] \dLG 32UN650-Wcafga \d ẋ


# @overload
# def initialize_from(
#     lookup_table: dict[str, ObjectType], __name__: str, **kwargs: Any
# ) -> ObjectType:
#     ...


# partial from type
@overload
def initialize_from(
    lookup_table: dict[str, type[ObjectType]],
    /,
    __name__: str,
    **kwargs: Any,
) -> ObjectType:
    ...


# partial from func
@overload
def initialize_from(
    lookup_table: dict[str, Callable[..., ReturnType]],
    /,
    __name__: str,
    **kwargs: Any,
) -> Callable[..., ReturnType]:
    ...


# partial from type
# @overload
# def initialize_from(
#     lookup_table: dict[str, Union[type[ObjectType], Callable[..., ReturnType]]],
#     /,
#     __name__: str,
#     **kwargs: Any,
# ) -> Union[ObjectType, Callable[..., ReturnType]]:
#     ...


def initialize_from(  # type: ignore[misc]
    lookup_table: Union[
        dict[str, type[ObjectType]],
        dict[str, Callable[..., ReturnType]],
        dict[str, Union[type[ObjectType], Callable[..., ReturnType]]],
        dict[str, Union[Callable[..., ReturnType], type[ObjectType]]],
    ],
    /,
    __name__: str,
    **kwargs: Any,
) -> Union[ObjectType, Callable[..., ReturnType]]:
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
    object
        The initialized class/function
    """
    obj = lookup_table[__name__]
    assert callable(obj), f"Looked up object {obj} not callable class/function."

    # check that obj is a class, but not metaclass or instance.
    if isinstance(obj, type) and not issubclass(obj, type):
        initialized_object: ObjectType = obj(**kwargs)
        return initialized_object
    # if it is function, fix kwargs
    initialized_callable: Callable[..., ReturnType] = partial(obj, **kwargs)  # type: ignore[assignment]
    return initialized_callable


def is_dunder(name: str) -> bool:
    r"""Check if name is a dunder method.

    Parameters
    ----------
    name: str

    Returns
    -------
    bool
    """
    return name.startswith("__") and name.endswith("__")


def is_partition(*partition: Collection, union: Optional[Sequence] = None) -> bool:
    r"""Check if partition is a valid partition of union."""
    if len(partition) == 1:
        return is_partition(*next(iter(partition)), union=union)

    sets = (set(p) for p in partition)
    part_union = set().union(*sets)

    if union is not None and part_union != set(union):
        return False
    return len(part_union) == sum(len(p) for p in partition)


def initialize_from_config(config: dict[str, Any]) -> Any:
    r"""Initialize a class from a dictionary.

    Parameters
    ----------
    config: dict[str, Any]

    Returns
    -------
    object
    """
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


def paths_exists(
    paths: Nested[Optional[PathType]],
    *,
    parent: Path = EmptyPath,
) -> bool:
    r"""Check whether the files exist.

    The input can be arbitrarily nested data-structure with `Path` in leaves.

    Parameters
    ----------
    paths: None | Path | Collection[Path] | Mapping[Any, None | Path | Collection[Path]]
    parent: Path = Path(),

    Returns
    -------
    bool
    """
    if isinstance(paths, str):
        return Path(paths).exists()
    if paths is None:
        return True
    if isinstance(paths, Mapping):
        return all(paths_exists(f, parent=parent) for f in paths.values())
    if isinstance(paths, Collection):
        return all(paths_exists(f, parent=parent) for f in paths)
    if isinstance(paths, Path):
        return (parent / paths).exists()

    raise ValueError(f"Unknown type for rawdata_file: {type(paths)}")