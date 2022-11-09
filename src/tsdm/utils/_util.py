r"""Utility functions.

TODO:  Module description
"""

# from __future__ import annotations


__all__ = [
    # Constants
    "NULL_VALUES",
    # Classes
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "get_function_args",
    "get_mandatory_argcount",
    "initialize_from_table",
    "initialize_from_config",
    "is_dunder",
    "is_keyword_only",
    "is_mandatory",
    "is_partition",
    "is_positional",
    "is_positional_only",
    "now",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
    "paths_exists",
    "prepend_path",
    "round_relative",
    "unflatten_dict",
]

import inspect
import os
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import datetime
from functools import partial
from importlib import import_module
from logging import getLogger
from pathlib import Path
from typing import Any, Final, Literal, Optional, overload

import numpy as np
from numpy.typing import NDArray
from torch import nn

from tsdm.utils.types import AnyTypeVar, Nested, ObjectVar, PathType, ReturnVar
from tsdm.utils.types.abc import HashableType

__logger__ = getLogger(__name__)
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
EMPTY = inspect.Parameter.empty
Kind = inspect._ParameterKind  # pylint: disable=protected-access

EMPTY_PATH: Final[Path] = Path()
r"""Constant: Blank path."""

# fmt: off
NULL_VALUES: Final[list[str]] = [
    "", "-", "?",
    "1.#IND", "+1.#IND", "-1.#IND", "1.#QNAN", "+1.#QNAN", "-1.#QNAN",
    "#N/A N/A",
    "#NA", "#Na", "#na", "<NA>", "<Na>", "<na>", "+NA", "-NA", "-na", "+na",
    "N/A", "n/a", "#N/A", "#n/a", "<N/A>", "<n/a>",  "+N/A", "-N/A", "-n/a", "+n/a",
    "NAN", "NaN", "nan", "-NAN", "-NaN", "-nan", "+NAN", "+NaN", "+nan",
    "#NAN", "#NaN", "#nan", "<NAN>", "<NaN>", "<nan>"
    "NONE", "None", "none", "#NONE", "#None", "#none", "<NONE>", "<None>", "<none>",
    "NULL", "Null", "null", "#NULL", "#Null", "#null", "<NULL>", "<Null>", "<null>",
    "UNKNOWN", "Unknown", "unknown", "UNKNOWN", "#Unknown", "#unknown", "<UNKNOWN>", "<Unknown>", "<unknown>",
]
r"""A list of common null value string represenations."""
# fmt: on


def variants(s: str | list[str]) -> list[str]:
    r"""Return all variants of a string."""
    if isinstance(s, str):
        cases = (
            lambda x: x.lower(),
            lambda x: x.capitalize(),
            lambda x: x.upper(),
        )
        decorations = (
            lambda x: x,
            lambda x: f"#{x}",
            lambda x: f"<{x}>",
            lambda x: f"+{x}",
            lambda x: f"-{x}",
        )
        return [deco(case(s)) for deco in decorations for case in cases]
    # return concatenation of all variants
    return [j for i in (variants(s_) for s_ in s) for j in i]


def pairwise_disjoint(sets: Iterable[set]) -> bool:
    r"""Check if sets are pairwise disjoint."""
    union = set().union(*sets)
    return len(union) == sum(len(s) for s in sets)


def pairwise_disjoint_masks(masks: Iterable[NDArray[np.bool_]]) -> bool:
    r"""Check if masks are pairwise disjoint."""
    return all(sum(masks) == 1)  # type: ignore[arg-type]


def apply_nested(
    nested: Nested[AnyTypeVar | None],
    kind: type[AnyTypeVar],
    func: Callable[[AnyTypeVar], ReturnVar],
) -> Nested[ReturnVar | None]:
    r"""Apply function to nested iterables of a given kind.

    Args:
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


def flatten_nested(nested: Any, kind: type[HashableType]) -> set[HashableType]:
    r"""Flatten nested iterables of a given kind."""
    if nested is None:
        return set()
    if isinstance(nested, kind):
        return {nested}
    if isinstance(nested, Mapping):
        return set.union(*(flatten_nested(v, kind) for v in nested.values()))
    if isinstance(nested, Iterable):
        return set.union(*(flatten_nested(v, kind) for v in nested))
    raise ValueError(f"{type(nested)} is not understood")


def flatten_dict(
    d: dict[str, Any],
    /,
    *,
    recursive: bool = True,
    join_fn: Callable[[Sequence[str]], str] = ".".join,
) -> dict[str, Any]:
    r"""Flatten dictionaries recursively."""
    result = {}
    for key, item in d.items():
        if isinstance(item, dict) and recursive:
            subdict = flatten_dict(item, recursive=True, join_fn=join_fn)
            for subkey, subitem in subdict.items():
                result[join_fn((key, subkey))] = subitem
        else:
            result[key] = item
    return result


def unflatten_dict(
    d: dict[str, Any],
    /,
    *,
    recursive: bool = True,
    split_fn: Callable[[str], Sequence[str]] = lambda s: s.split(".", maxsplit=1),
) -> dict[str, Any]:
    r"""Unflatten dictionaries recursively."""
    result: dict[str, Any] = {}
    for key, item in d.items():
        split = split_fn(key)
        result.setdefault(split[0], {})
        if len(split) > 1 and recursive:
            assert len(split) == 2
            subdict = unflatten_dict(
                {split[1]: item}, recursive=recursive, split_fn=split_fn
            )
            result[split[0]] |= subdict
        else:
            result[split[0]] = item
    return result


def round_relative(x: np.ndarray, decimals: int = 2) -> np.ndarray:
    r"""Round to relative precision."""
    order = np.where(x == 0, 0, np.floor(np.log10(x)))
    digits = decimals - order
    rounded = np.rint(x * 10**digits)
    return np.true_divide(rounded, 10**digits)


def now() -> str:
    r"""Return current time in iso format."""
    return datetime.now().isoformat(timespec="seconds")


def deep_dict_update(d: dict, new_kvals: Mapping) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References
    ----------
    - https://stackoverflow.com/a/30655448/9318372
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


def deep_kval_update(d: dict, **new_kvals: Any) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    References
    ----------
    - https://stackoverflow.com/a/30655448/9318372
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
    keep_none: bool = True,
) -> Nested[Optional[Path]]:
    r"""Prepends path to all files in nested iterable.

    If `keep_none=True`, then `None` values are kept, else they are replaced by
    ``parent``.
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


def paths_exists(
    paths: Nested[Optional[PathType]],
    *,
    parent: Path = EMPTY_PATH,
) -> bool:
    r"""Check whether the files exist.

    The input can be arbitrarily nested data-structure with `Path` in leaves.
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


def initialize_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize `nn.Module` from config object."""
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


@overload
def initialize_from_table(  # type: ignore[misc]
    lookup_table: dict[str, type[ObjectVar]], /, __name__: str, **kwargs: Any
) -> ObjectVar:
    ...


@overload
def initialize_from_table(
    lookup_table: dict[str, Callable[..., ReturnVar]], /, __name__: str, **kwargs: Any
) -> Callable[..., ReturnVar]:
    ...


def initialize_from_table(lookup_table, /, __name__, **kwargs):
    r"""Lookup class/function from dictionary and initialize it.

    Roughly equivalent to:

    .. code-block:: python

        obj = lookup_table[__name__]
        if isclass(obj):
            return obj(**kwargs)
        return partial(obj, **kwargs)
    """
    obj = lookup_table[__name__]
    assert callable(obj), f"Looked up object {obj} not callable class/function."

    # check that obj is a class, but not metaclass or instance.
    if isinstance(obj, type) and not issubclass(obj, type):
        initialized_object = obj(**kwargs)
        return initialized_object

    # if it is function, fix kwargs
    initialized_callable = partial(obj, **kwargs)
    return initialized_callable


def is_dunder(name: str) -> bool:
    r"""Check if name is a dunder method."""
    return name.isidentifier() and name.startswith("__") and name.endswith("__")


def is_partition(*partition: Collection, union: Optional[Sequence] = None) -> bool:
    r"""Check if partition is a valid partition of union."""
    if len(partition) == 1:
        return is_partition(*next(iter(partition)), union=union)

    sets = (set(p) for p in partition)
    part_union = set().union(*sets)

    if union is not None and part_union != set(union):
        return False
    return len(part_union) == sum(len(p) for p in partition)


def is_mandatory(p: inspect.Parameter, /) -> bool:
    r"""Check if parameter is mandatory."""
    return p.default is inspect.Parameter.empty and p.kind not in (
        VAR_POSITIONAL,
        VAR_KEYWORD,
    )


def is_positional(p: inspect.Parameter, /) -> bool:
    r"""Check if parameter is positional."""
    return p.kind in (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL)


def is_positional_only(p: inspect.Parameter, /) -> bool:
    """Check if parameter is positional only."""
    return p.kind in (POSITIONAL_ONLY, VAR_POSITIONAL)


def is_keyword_only(p: inspect.Parameter, /) -> bool:
    """Check if parameter is keyword only."""
    return p.kind in (KEYWORD_ONLY, VAR_KEYWORD)


def get_mandatory_argcount(f: Callable[..., Any]) -> int:
    r"""Get the number of mandatory arguments of a function."""
    sig = inspect.signature(f)
    return sum(is_mandatory(p) for p in sig.parameters.values())


def get_function_args(
    f: Callable[..., Any],
    mandatory: Optional[bool] = None,
    kinds: Optional[str | Kind | list[Kind]] = None,
) -> list[inspect.Parameter]:
    r"""Filter function parameters by kind and optionality."""

    def get_kinds(s: str | Kind) -> set[Kind]:
        if isinstance(s, Kind):
            return {s}

        match s.lower():
            case "p" | "positional":
                return {POSITIONAL_ONLY, VAR_POSITIONAL}
            case "k" | "keyword":
                return {KEYWORD_ONLY, VAR_KEYWORD}
            case "v" | "var":
                return {VAR_POSITIONAL, VAR_KEYWORD}
            case "po" | "positional_only":
                return {POSITIONAL_ONLY}
            case "ko" | "keyword_only":
                return {KEYWORD_ONLY}
            case "pk" | "positional_or_keyword":
                return {POSITIONAL_OR_KEYWORD}
            case "vp" | "var_positional":
                return {VAR_POSITIONAL}
            case "vk" | "var_keyword":
                return {VAR_KEYWORD}
        raise ValueError(f"Unknown kind {s}")

    match kinds:
        case None:
            allowed_kinds = set(Kind)
        case str() | Kind():
            allowed_kinds = get_kinds(kinds)
        case Sequence():
            allowed_kinds = set().union(*map(get_kinds, kinds))
        case _:
            raise ValueError(f"Unknown type for kinds: {type(kinds)}")

    sig = inspect.signature(f)
    params = list(sig.parameters.values())

    if mandatory is None:
        return [p for p in params if p.kind in allowed_kinds]

    return [
        p for p in params if is_mandatory(p) is mandatory and p.kind in allowed_kinds
    ]
