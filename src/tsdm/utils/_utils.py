r"""Utility functions.

TODO:  Module description
"""

__all__ = [
    # Classes
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "initialize_from_config",
    "is_dunder",
    "is_zipfile",
    "now",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
    "paths_exists",
    "prepend_path",
    "repackage_zip",
    "round_relative",
    "unflatten_dict",
]

import os
import shutil
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from datetime import datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas
from numpy.typing import NDArray
from pandas import Series
from torch import nn
from tqdm.autonotebook import tqdm
from typing_extensions import Any, Literal, Optional, cast, overload

from tsdm.constants import BOOLEAN_PAIRS
from tsdm.types.aliases import Nested, NestedDict, NestedMapping, PathLike
from tsdm.types.variables import HashableType, key_other_var as K2, key_var as K


def variants(s: str | list[str]) -> list[str]:
    r"""Return all variants of a string."""
    if isinstance(s, str):
        cases: list[Callable[[str], str]] = [
            lambda x: x.lower(),
            lambda x: x.capitalize(),
            lambda x: x.upper(),
        ]
        decorations: list[Callable[[str], str]] = [
            lambda x: x,
            lambda x: f"#{x}",
            lambda x: f"<{x}>",
            lambda x: f"+{x}",
            lambda x: f"-{x}",
        ]
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


def flatten_nested(
    nested: Any, /, *, leaf_type: type[HashableType]
) -> set[HashableType]:
    r"""Flatten nested iterables of a given kind."""
    if nested is None:
        return set()
    if isinstance(nested, leaf_type):
        return {nested}
    if isinstance(nested, Mapping):
        return set.union(
            *(flatten_nested(v, leaf_type=leaf_type) for v in nested.values())
        )
    if isinstance(nested, Iterable):
        return set.union(*(flatten_nested(v, leaf_type=leaf_type) for v in nested))
    raise ValueError(f"{type(nested)} is not understood")


def is_flattened(
    d: Mapping, /, *, key_type: type = object, val_type: type = Mapping
) -> bool:
    r"""Check if mapping is flattened."""
    return all(
        isinstance(key, key_type) and not isinstance(val, val_type)
        for key, val in d.items()
    )


def flatten_dict(
    d: NestedMapping[K, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = ".".join,  # type: ignore[assignment]
    split_fn: Callable[[K2], Iterable[K]] = lambda s: s.split("."),  # type: ignore[attr-defined]
) -> dict[K2, Any]:
    r"""Flatten dictionaries recursively.

    Args:
        d: dictionary to flatten
        recursive: whether to flatten recursively. If `recursive` is an integer,
            then it flattens recursively up to `recursive` levels.
        join_fn: function to join keys
        split_fn: function to split keys

    Examples:
        Using ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``
        will combine string keys like ``"a"`` and ``"b"`` into ``"a.b"``.

        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {'a.b': 1, 'a.c': 2}

        >>> flatten_dict({"a": {"b": {"x": 2}, "c": 2}})
        {'a.b.x': 2, 'a.c': 2}

        >>> flatten_dict({"a": {"b": 1, "c": 2}}, recursive=False)
        {'a': {'b': 1, 'c': 2}}

        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will combine
        keys like ``("a", "b")`` and ``("a", "c")`` into ``("a", "b", "c")``.

        >>> flatten_dict({"a": {"b": 1, "c": 2}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 'b'): 1, ('a', 'c'): 2}

        >>> flatten_dict({"a": {"b": {"x": 2}, "c": 2}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 'b', 'x'): 2, ('a', 'c'): 2}

        >>> flatten_dict({"a": {17: "foo", 18: "bar"}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 17): 'foo', ('a', 18): 'bar'}

        When trying to flatten a partially flattened dictionary, setting recursive=<int>.

        >>> flatten_dict({"a": {(1, True): "foo", (2, False): "bar"}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', (1, True)): 'foo', ('a', (2, False)): 'bar'}

        >>> flatten_dict({"a": {(1, True): "foo", (2, False): "bar"}}, join_fn=tuple, split_fn=lambda x: x, recursive=1)
        {('a', 1, True): 'foo', ('a', 2, False): 'bar'}
    """
    if not recursive:
        return cast(dict[K2, Any], dict(d))

    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K2, Any] = {}
    for key, item in d.items():
        if isinstance(item, Mapping):
            for subkey, subitem in flatten_dict(
                item,
                recursive=recursive,
                join_fn=join_fn,
                split_fn=split_fn,
            ).items():
                new_key = join_fn((key, *split_fn(subkey)))
                result[new_key] = subitem
        else:
            new_key = join_fn((key,))
            result[new_key] = item
    return result


def unflatten_dict(
    d: Mapping[K2, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = ".".join,  # type: ignore[assignment]
    split_fn: Callable[[K2], Iterable[K]] = lambda s: s.split("."),  # type: ignore[attr-defined]
) -> NestedDict[K, Any]:
    r"""Unflatten dictionaries recursively.

    Examples:
        Using ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``
        will split up string keys like ``"a.b.c"`` into ``{"a": {"b": {"c": ...}}}``.

        >>> unflatten_dict({'a.b': 1, 'a.c': 2})
        {'a': {'b': 1, 'c': 2}}

        >>> unflatten_dict({'a.b': 1, 'a.c': 2}, recursive=False)
        {'a.b': 1, 'a.c': 2}

        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will split up
        keys like ``("a", "b", "c")`` into ``{"a": {"b": {"c": ...}}}``.

        >>> unflatten_dict({('a', 17): 'foo', ('a', 18): 'bar'}, join_fn=tuple, split_fn=lambda x: x)
        {'a': {17: 'foo', 18: 'bar'}}
    """
    if not recursive:
        return cast(dict[K, Any], dict(d))

    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K, Any] = {}
    for key, item in d.items():
        outer_key, *inner_keys = split_fn(key)
        if inner_keys:
            assert isinstance(d, Mapping), "d must be a Mapping!"
            result.setdefault(outer_key, {})
            result[outer_key] |= unflatten_dict(
                {join_fn(inner_keys): item},
                recursive=recursive,
                split_fn=split_fn,
                join_fn=join_fn,
            )
        else:
            result[outer_key] = item
    return result


def round_relative(x: np.ndarray, /, *, decimals: int = 2) -> np.ndarray:
    r"""Round to relative precision."""
    order = np.where(x == 0, 0, np.floor(np.log10(x)))
    digits = decimals - order
    rounded = np.rint(x * 10**digits)
    return np.true_divide(rounded, 10**digits)


def now() -> str:
    r"""Return current time in iso format."""
    return datetime.now().isoformat(timespec="seconds")


def deep_dict_update(d: dict, new_kvals: Mapping, /) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References:
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


def deep_kval_update(d: dict, /, **new_kvals: Any) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    References:
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
    files: Mapping[K, PathLike],
    parent: Path,
    /,
    *,
    keep_none: bool = False,
) -> dict[K, Path]: ...
@overload
def prepend_path(
    files: list[PathLike],
    parent: Path,
    /,
    *,
    keep_none: bool = False,
) -> list[Path]: ...
@overload
def prepend_path(
    files: PathLike,
    parent: Path,
    /,
    *,
    keep_none: bool = False,
) -> Path: ...
@overload
def prepend_path(
    files: Nested[PathLike],
    parent: Path,
    /,
    *,
    keep_none: bool = False,
) -> Nested[Path]: ...
@overload
def prepend_path(
    files: Nested[Optional[PathLike]],
    parent: Path,
    /,
    *,
    keep_none: Literal[False] = False,
) -> Nested[Path]: ...
@overload
def prepend_path(
    files: Nested[Optional[PathLike]],
    parent: Path,
    /,
    *,
    keep_none: Literal[True] = True,
) -> Nested[Optional[Path]]: ...
def prepend_path(
    files: Nested[Optional[PathLike]],
    parent: Path,
    /,
    *,
    keep_none: bool = True,
) -> Nested[Optional[Path]]:
    r"""Prepends the given path to all files in nested iterable.

    If `keep_none=True`, then `None` values are kept, else they are replaced by `parent`.
    """
    # TODO: change it to apply_nested in python 3.10
    match files:
        case None:
            return None if keep_none else parent
        case str() | Path() | os.PathLike() as path:
            return parent / Path(path)
        case Mapping() as mapping:
            return {
                k: prepend_path(v, parent, keep_none=keep_none) for k, v in mapping.items()  # type: ignore[arg-type]
            }
        case Collection() as coll:
            # TODO https://github.com/python/mypy/issues/11615
            return [prepend_path(f, parent, keep_none=keep_none) for f in coll]  # type: ignore[arg-type]
        case _:
            raise TypeError(f"Unsupported type: {type(files)}")


def paths_exists(paths: Nested[Optional[PathLike]], /) -> bool:
    r"""Check whether the files exist.

    The input can be arbitrarily nested data-structure with `Path` in leaves.
    """
    match paths:
        case None:
            return True
        case str() as string:
            return Path(string).exists()
        case Path() as path:
            return path.exists()
        case Mapping() as mapping:
            return all(paths_exists(f) for f in mapping.values())
        case Iterable() as iterable:
            return all(paths_exists(f) for f in iterable)
        case _:
            raise TypeError(f"Unknown type for rawdata_file: {type(paths)}")


def initialize_from_config(config: dict[str, Any], /) -> nn.Module:
    r"""Initialize `nn.Module` from a config object."""
    assert "__name__" in config, "__name__ not found in dict!"
    assert "__module__" in config, "__module__ not found in dict!"
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


def is_dunder(name: str, /) -> bool:
    r"""Check if the name is a dunder method."""
    return name.isidentifier() and name.startswith("__") and name.endswith("__")


def is_zipfile(path: Path, /) -> bool:
    r"""Return `True` if the file is a zipfile."""
    try:
        with ZipFile(path):
            return True
    except (BadZipFile, IsADirectoryError):
        return False


def get_uniques(series: Series, /, *, ignore_nan: bool = True) -> Series:
    r"""Return unique values, excluding nan."""
    if ignore_nan:
        mask = pandas.notna(series)
        series = series[mask]
    return Series(series.unique())


def series_is_boolean(series: Series, /, *, uniques: Optional[Series] = None) -> bool:
    r"""Test if 'string' series could possibly be boolean."""
    assert pandas.api.types.is_string_dtype(series), "Series must be 'string' dtype!"
    values = get_uniques(series) if uniques is None else uniques

    if len(values) == 0 or len(values) > 2:
        return False
    return any(
        set(values.str.lower()) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS
    )


def series_to_boolean(s: Series, uniques: Optional[Series] = None, /) -> Series:
    r"""Convert Series to nullable boolean."""
    assert pandas.api.types.is_string_dtype(s), "Series must be 'string' dtype!"
    mask = pandas.notna(s)
    values = get_uniques(s[mask]) if uniques is None else uniques
    mapping = next(
        set(values.str.lower()) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS
    )
    s = s.copy()
    s[mask] = s[mask].map(mapping)
    return s.astype(pandas.BooleanDtype())


def series_numeric_is_boolean(s: Series, uniques: Optional[Series] = None, /) -> bool:
    r"""Test if 'numeric' series could possibly be boolean."""
    assert pandas.api.types.is_numeric_dtype(s), "Series must be 'numeric' dtype!"
    values = get_uniques(s) if uniques is None else uniques
    if len(values) == 0 or len(values) > 2:
        return False
    return any(set(values) <= set(bool_pair) for bool_pair in BOOLEAN_PAIRS)


def series_is_int(s: Series, uniques: Optional[Series] = None, /) -> bool:
    r"""Check whether float encoded column holds only integers."""
    assert pandas.api.types.is_float_dtype(s), "Series must be 'float' dtype!"
    values = get_uniques(s) if uniques is None else uniques
    return cast(bool, values.apply(float.is_integer).all())


def repackage_zip(path: PathLike, /) -> None:
    """Remove the leading directory from a zip file."""
    original_path = Path(path)

    if not is_zipfile(original_path):
        warnings.warn(f"{original_path} is not a zip file.", stacklevel=2)
        return

    # guard clause: check if requirements are met
    with ZipFile(original_path, "r") as original_archive:
        contents = original_archive.namelist()
        top = contents[0]

        requirements = (
            top.endswith("/")
            # zip file name must match top directory name
            and original_path.stem == top[:-1]
            # all items must start with top directory name
            and all(item.startswith(top) for item in contents)
        )

        if not requirements:
            print(f"Skipping repackage_zip for {original_path}.")
            return

    # create a temporary directory
    with TemporaryDirectory() as temp_dir:
        # move the zip file to the temporary directory
        temp_path = Path(temp_dir) / original_path.name
        shutil.move(original_path, temp_path)
        # create a new zipfile with the modified contents:
        with ZipFile(temp_path) as old_archive, ZipFile(path, "w") as new_archive:
            contents = old_archive.namelist()
            top = contents[0]
            for item in tqdm(contents[1:], desc="Repackaging zip file"):
                _, new_name = item.split(top, 1)
                new_archive.writestr(new_name, old_archive.read(item))
