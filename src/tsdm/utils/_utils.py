r"""Utility functions."""

__all__ = [
    # Classes
    # Functions
    "axes_to_tuple",
    "deep_dict_update",
    "deep_kval_update",
    "dims_to_list",
    "flatten_dict",
    "flatten_nested",
    "initialize_from_config",
    "joint_keys",
    "last",
    "timedelta",
    "timestamp",
    "now",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
    "paths_exists",
    "prepend_path",
    "repackage_zip",
    "replace",
    "round_relative",
    "shape_to_tuple",
    "size_to_tuple",
    "unflatten_dict",
]

import os
import shutil
import warnings
from collections import deque
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    Reversible,
    Sequence,
)
from datetime import datetime
from functools import wraps
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray
from pandas import Timedelta, Timestamp
from pandas.api.typing import NaTType
from torch import nn
from tqdm.autonotebook import tqdm
from typing_extensions import Any, Literal, Optional, cast, overload

from tsdm.constants import EMPTY_MAP
from tsdm.testing._testing import is_dunder, is_zipfile
from tsdm.types.aliases import (
    Axis,
    Dims,
    FilePath,
    MaybeWrapped,
    Nested,
    NestedDict,
    NestedMapping,
    Shape,
    Size,
)
from tsdm.types.variables import K2, HashableType, K, T


@wraps(Timedelta)
def timedelta(value: Any = ..., unit: Optional[str] = None, **kwargs: Any) -> Timedelta:
    r"""Utility function that ensures that the constructor does not return NaT."""
    td = (
        Timedelta(unit=unit, **kwargs)
        if value is Ellipsis
        else Timedelta(value, unit=unit, **kwargs)
    )
    if isinstance(td, NaTType):
        raise TypeError("Constructor returned NaT")
    return td


@wraps(Timestamp)
def timestamp(value: Any = ..., **kwargs: Any) -> Timestamp:
    r"""Utility function that ensures that the constructor does not return NaT."""
    ts = Timestamp(**kwargs) if value is Ellipsis else Timestamp(value, **kwargs)
    if isinstance(ts, NaTType):
        raise TypeError("Constructor returned NaT")
    return ts


def axes_to_tuple(axes: Axis, *, ndim: int) -> tuple[int, ...]:
    """Convert axes to tuple.

    Note:
        - `ndarray.mean(axis=None)` contracts over all axes, returns scalar.
        - `ndarray.mean(axis=[])`   contracts over no axes, returns copy.
        - `ndarray.mean(axis=k)`    contracts over the k-th axis.
        - `ndarray.mean(axis=tuple(range(ndim)))` contracts over all axes, returns scalar.
    """
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        return (axes,)
    return tuple(axes)


def dims_to_list(dims: Dims, *, ndim: int) -> list[int]:
    """Convert dimensions to list.

    Note:
        - `tensor.mean(dim=None)` contracts over all dims, returns 1d-tensor (1 element).
        - `tensor.mean(dim=[])`   contracts over all dims, returns 1d-tensor (1 element).
        - `tensor.mean(dim=k)`    contracts over the k-th dimension.
        - `tensor.mean(dim=list(range(ndim)))` contracts over all dims, returns 1d-tensor (1 element).
    """
    if dims is None:
        return list(range(ndim))
    if isinstance(dims, int):
        return [dims]
    return list(dims)


def size_to_tuple(size: Size) -> tuple[int, ...]:
    """Convert size to tuple.

    Note:
        - `np.random.normal(size=None)` produces a scalar.
        - `np.random.normal(size=())`   produces a 0d-array (1 element).
        - `np.random.normal(size=k)`    produces a 1d-array (k elements).
    """
    if isinstance(size, int):
        return (size,)
    return tuple(size)


def shape_to_tuple(shape: Shape) -> tuple[int, ...]:
    """Convert shape to tuple.

    Note:
        - `np.ones(shape=None)` produces a 0d-array (1 element).
        - `np.ones(shape=())`   produces a 0d-array (1 element).
        - `np.ones(shape=k)`    produces a 1d-array (k elements).
    """
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def last(iterable: Iterable[T], /) -> T:
    r"""Return the last element of an `Iterable`.

    Raises:
        ValueError: if iterable is empty.
    """
    match iterable:
        case Sequence() as seq:
            try:
                return seq[-1]
            except IndexError as exc:
                raise ValueError("Sequence is empty!") from exc
        case Reversible() as rev:
            try:
                return next(reversed(rev))
            except StopIteration as exc:
                raise ValueError("Reversible is empty!") from exc
        case _:
            try:
                return deque(iterable, maxlen=1).pop()
            except IndexError as exc:
                raise ValueError("Iterable is empty!") from exc


def replace(s: str, mapping: Mapping[str, str] = EMPTY_MAP, /, **strings: str) -> str:
    r"""Replace multiple substrings via dict.

    References:
        https://stackoverflow.com/a/64500851
    """
    replacements = dict(mapping, **strings)
    return last((s := s.replace(x, y) for x, y in replacements.items()))


def variants(s: str | list[str], /) -> list[str]:
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


def pairwise_disjoint(sets: Iterable[set], /) -> bool:
    r"""Check if sets are pairwise disjoint."""
    union = set().union(*sets)
    return len(union) == sum(len(s) for s in sets)


def pairwise_disjoint_masks(masks: Iterable[NDArray[np.bool_]], /) -> bool:
    r"""Check if masks are pairwise disjoint."""
    return all(sum(masks) == 1)  # type: ignore[arg-type]


def flatten_nested(
    nested: Any, /, *, leaf_type: type[HashableType]
) -> set[HashableType]:
    r"""Flatten nested iterables of a given kind."""
    match nested:
        case None:
            return set()
        case leaf_type() as leaf:  # type: ignore[misc]
            return {leaf}  # type: ignore[unreachable]
        case Mapping() as mapping:
            return set.union(
                *(flatten_nested(v, leaf_type=leaf_type) for v in mapping.values())
            )
        case Iterable() as iterable:
            return set.union(
                *(flatten_nested(v, leaf_type=leaf_type) for v in iterable)
            )
        case _:
            raise TypeError(f"{type(nested)=} not understood")


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

        >>> flatten_dict(
        ...     {"a": {"b": {"x": 2}, "c": 2}}, join_fn=tuple, split_fn=lambda x: x
        ... )
        {('a', 'b', 'x'): 2, ('a', 'c'): 2}

        >>> flatten_dict({"a": {17: "foo", 18: "bar"}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 17): 'foo', ('a', 18): 'bar'}

        When trying to flatten a partially flattened dictionary, setting recursive=<int>.

        >>> flatten_dict(
        ...     {"a": {(1, True): "foo", (2, False): "bar"}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {('a', (1, True)): 'foo', ('a', (2, False)): 'bar'}

        >>> flatten_dict(
        ...     {"a": {(1, True): "foo", (2, False): "bar"}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ...     recursive=1,
        ... )
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

        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {'a': {'b': 1, 'c': 2}}

        >>> unflatten_dict({"a.b": 1, "a.c": 2}, recursive=False)
        {'a.b': 1, 'a.c': 2}

        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will split up
        keys like ``("a", "b", "c")`` into ``{"a": {"b": {"c": ...}}}``.

        >>> unflatten_dict(
        ...     {("a", 17): "foo", ("a", 18): "bar"}, join_fn=tuple, split_fn=lambda x: x
        ... )
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
        - https://stackoverflow.com/a/30655448
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in new_kvals.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:  # if value is not None or not safe:
            d[key] = value
    return d


def deep_kval_update(d: dict, /, **new_kvals: Any) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    References:
        - https://stackoverflow.com/a/30655448
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
    files: Mapping[K, FilePath], parent: Path, /, *, keep_none: bool = ...
) -> dict[K, Path]: ...
@overload
def prepend_path(
    files: list[FilePath], parent: Path, /, *, keep_none: bool = ...
) -> list[Path]: ...
@overload
def prepend_path(
    files: FilePath, parent: Path, /, *, keep_none: bool = ...
) -> Path: ...
@overload
def prepend_path(
    files: Nested[FilePath], parent: Path, /, *, keep_none: bool = ...
) -> Nested[Path]: ...
@overload
def prepend_path(
    files: Nested[Optional[FilePath]], parent: Path, /, *, keep_none: Literal[False]
) -> Nested[Path]: ...
@overload
def prepend_path(
    files: Nested[Optional[FilePath]], parent: Path, /, *, keep_none: Literal[True]
) -> Nested[Optional[Path]]: ...
def prepend_path(files, parent, /, *, keep_none=True):
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
                k: prepend_path(v, parent, keep_none=keep_none)
                for k, v in mapping.items()
            }
        case Collection() as coll:
            return [prepend_path(f, parent, keep_none=keep_none) for f in coll]
        case _:
            raise TypeError(f"Unsupported type: {type(files)}")


def paths_exists(paths: Nested[Optional[FilePath]], /) -> bool:
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


def repackage_zip(path: FilePath, /) -> None:
    r"""Remove the leading directory from a zip file."""
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


def joint_keys(*mappings: Mapping[T, Any]) -> set[T]:
    r"""Find joint keys in a collection of Mappings."""
    # NOTE: `.keys()` is necessary for working with `pandas.Series` and `pandas.DataFrame`.
    return set.intersection(*map(set, (d.keys() for d in mappings)))


def unpack_maybewrapped(x: MaybeWrapped[T], /, *, step: int) -> T:
    r"""Unpack wrapped values."""
    if callable(x):
        try:
            return x(step)  # type: ignore[call-arg]
        except TypeError:
            return x()  # type: ignore[call-arg]

    return x


def transpose_list_of_dicts(lst: Iterable[dict[K, T]], /) -> dict[K, list[T]]:
    r"""Fast way to 'transpose' a list of dictionaries.

    Assumptions:
        - all dictionaries have the same keys
        - the keys are always in the same order
        - at least one item in the input
        - can iterate multiple times over lst

    Example:
        >>> list_of_dicts = [
        ...     {"name": "Alice", "age": 30},
        ...     {"name": "Bob", "age": 25},
        ...     {"name": "Charlie", "age": 35},
        ... ]
        >>> transpose_list_of_dicts(list_of_dicts)
        {'name': ('Alice', 'Bob', 'Charlie'), 'age': (30, 25, 35)}
    """
    return dict(
        zip(
            next(iter(lst)),
            zip(*(d.values() for d in lst), strict=True),
            strict=True,
        )
    )
