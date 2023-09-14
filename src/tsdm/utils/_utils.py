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
    "is_partition",
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

import logging
import os
import shutil
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Optional, cast, overload
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas
from numpy.typing import NDArray
from pandas import Series
from torch import nn
from tqdm.autonotebook import tqdm

from tsdm.constants import BOOLEAN_PAIRS
from tsdm.types.abc import HashableType
from tsdm.types.aliases import Nested, PathLike
from tsdm.types.variables import key_var as K

__logger__ = logging.getLogger(__name__)


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
    r"""Prepends path to all files in nested iterable.

    If `keep_none=True`, then `None` values are kept, else they are replaced by `parent`.
    """
    # TODO: change it to apply_nested in python 3.10

    if files is None:
        return None if keep_none else parent
    if isinstance(files, str | Path | os.PathLike):
        return parent / Path(files)
    if isinstance(files, Mapping):
        return {
            k: prepend_path(v, parent, keep_none=keep_none) for k, v in files.items()  # type: ignore[arg-type]
        }
    # TODO https://github.com/python/mypy/issues/11615
    if isinstance(files, Collection):
        return [prepend_path(f, parent, keep_none=keep_none) for f in files]  # type: ignore[arg-type]
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

    raise ValueError(f"Unknown type for rawdata_file: {type(paths)}")


def initialize_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize `nn.Module` from config object."""
    assert "__name__" in config, "__name__ not found in dict!"
    assert "__module__" in config, "__module__ not found in dict!"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


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


def is_zipfile(path: Path) -> bool:
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
            __logger__.info("%s: skipping repackage_zip.", original_path)
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
