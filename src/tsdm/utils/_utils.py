r"""Utility functions."""

__all__ = [
    # Classes
    # Functions
    "normalize_axes",
    "deep_dict_update",
    "normalize_dimarg",
    "flatten_dict",
    "flatten_nested",
    "last",
    "timedelta",
    "timestamp",
    "nested_paths_exist",
    "repackage_zip",
    "replace",
    "unflatten_dict",
]

import logging
import shutil
import warnings
from collections import deque
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping as MutMap,
    Reversible,
    Sequence,
)
from copy import deepcopy
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, cast, overload
from zipfile import ZipFile

from pandas import Timedelta, Timestamp
from pandas.api.typing import NaTType
from tqdm.auto import tqdm

from tsdm.constants import EMPTY_MAP
from tsdm.testing import is_zipfile
from tsdm.types.aliases import (
    Axis,
    FilePath,
    Nested,
    NestedDict,
    NestedMapping,
)


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


def normalize_axes(axes: Axis, *, ndim: int) -> tuple[int, ...]:
    r"""Convert axes to tuple.

    Note:
        - `ndarray.mean(axis=None)` contracts over all axes, returns scalar.
        - `ndarray.mean(axis=[])`   contracts over no axes, returns copy.
        - `ndarray.mean(axis=k)`    contracts over the k-th axis.
        - `ndarray.mean(axis=tuple(range(ndim)))` contracts over all axes, returns scalar.
    """
    match axes:
        case None:
            return tuple(range(ndim))
        case int():
            return (axes % ndim,)
        case [*items]:
            return tuple(ax % ndim for ax in items)
        case _:
            raise TypeError(f"Unknown type for axes: {type(axes)}")


def normalize_dimarg(dims: int | list[int] | None, *, ndim: int) -> list[int]:
    r"""Convert dimensions to list.

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


def last[T](iterable: Iterable[T], /) -> T:
    r"""Return the last element of an `Iterable`.

    Raises:
        ValueError: if iterable is empty.
    """
    match iterable:
        # fast-path for sequences
        case Sequence() as seq:
            try:
                return seq[-1]
            except IndexError as exc:
                raise ValueError("Sequence is empty!") from exc

        # fast-path for reversible iterables
        case Reversible() as rev:
            try:
                return next(reversed(rev))
            except StopIteration as exc:
                raise ValueError("Reversible is empty!") from exc

        # fallback for general iterables
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
    return last(s := s.replace(x, y) for x, y in replacements.items())


def flatten_nested[H](nested: Any, /, *, leaf_type: type[H]) -> set[H]:
    r"""Flatten nested iterables of a given kind."""
    match nested:
        case None:
            return set()
        case leaf if isinstance(leaf, leaf_type):
            return {leaf}
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


@overload
def flatten_dict(
    d: NestedMapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> dict[str, Any]: ...
@overload
def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2],
    split_fn: Callable[[K2], Iterable[K]],
    recursive: bool | int = ...,
) -> dict[K2, Any]: ...
def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # noqa: B008
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # noqa: B008
    recursive: bool | int = True,
) -> dict[K2, Any]:
    r"""Flatten dictionaries recursively.

    Args:
        d: dictionary to flatten
        recursive: whether to flatten recursively.
            If `recursive` is an integer, flattens that many levels.
        join_fn: function to join keys
            Defaults to ``'.'.join``, implicitly assuming that all keys are strings.
        split_fn: function to split keys
            Defaults to ``str.split('.')``, implicitly assuming that all keys are strings.

    Example: flattening with string keys.
        When ``join_fn`` and ``split_fn`` are not provided, they default to
        ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``,
        implicitly assuming that all keys are strings.
        This will combine string keys like ``"a"`` and ``"b"`` into ``"a.b"``.

        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {'a.b': 1, 'a.c': 2}

        >>> flatten_dict({"a": {"b": {"x": 2}, "c": 2}})
        {'a.b.x': 2, 'a.c': 2}

    Example: flattening with custom key functions.
        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will combine
        keys like ``("a", "b")`` and ``("a", "c")`` into ``("a", "b", "c")``.
        This choice works for arbitrary key types.

        >>> flatten_dict({"a": {"b": 1, "c": 2}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 'b'): 1, ('a', 'c'): 2}

        >>> flatten_dict(
        ...     {"a": {"b": {"x": 2}, "c": 2}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {('a', 'b', 'x'): 2, ('a', 'c'): 2}

    Example: partial flattening with ``recursive``.
        >>> flatten_dict({"a": {"i": {"x": 0}, "b": {"y": 1}}})
        {'a.i.x': 0, 'a.b.y': 1}

        >>> flatten_dict(
        ...     {"a": {"i": {"x": 0}, "b": {"y": 1}}},
        ...     recursive=2,
        ... )
        {'a.i': {'x': 0}, 'a.b': {'y': 1}}

        >>> flatten_dict(
        ...     {"a": {"i": {"x": 0}, "b": {"y": 1}}},
        ...     recursive=1,
        ... )
        {'a': {'i': {'x': 0}, 'b': {'y': 1}}}
    """
    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K2, Any] = {}
    for key, item in d.items():
        if recursive and isinstance(item, Mapping):
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


@overload
def unflatten_dict(
    d: Mapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> NestedDict[str, Any]: ...
@overload
def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2],
    split_fn: Callable[[K2], Iterable[K]],
    recursive: bool | int = ...,
) -> NestedDict[K, Any]: ...
def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # noqa: B008
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # noqa: B008
) -> NestedDict[K, Any]:
    r"""Unflatten dictionaries recursively.

    Example: Unflattening with string keys.
        When ``join_fn`` and ``split_fn`` are not provided, they default to
        ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``,
        implicitly assuming that all keys are strings.
        This will split up keys like ``"a.b"`` into ``{"a": {"b": ...}}``.
        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {'a': {'b': 1, 'c': 2}}

    Example: unflattening with custom join function.
        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will split up
        keys like ``("a", "b", "c")`` into ``{"a": {"b": {"c": ...}}}``.
        >>> unflatten_dict(
        ...     {("a", 17): "foo", ("a", 18): "bar"},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {'a': {17: 'foo', 18: 'bar'}}

    Example: partial unflattening with ``recursive``.
        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1})
        {'a': {'b': {'c': {'d': 0}}, 'x': {'y': {'z': 1}}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=2)
        {'a': {'b': {'c.d': 0}, 'x': {'y.z': 1}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=1)
        {'a': {'b.c.d': 0, 'x.y.z': 1}}
    """
    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K, Any] = {}
    for key, item in d.items():
        outer_key, *inner_keys = split_fn(key)
        if inner_keys:
            result.setdefault(outer_key, {})
            if not isinstance(result[outer_key], dict):
                raise KeyError(f"Key conflict at {outer_key}! Cannot unflatten.")
            if recursive:
                result[outer_key] |= unflatten_dict(
                    {join_fn(inner_keys): item},
                    recursive=recursive,
                    split_fn=split_fn,
                    join_fn=join_fn,
                )
            else:
                result[outer_key] |= {join_fn(inner_keys): item}
        elif outer_key in result:
            raise KeyError(f"Key conflict at {outer_key}! Cannot unflatten.")
        else:
            result[outer_key] = item
    return result


def deep_dict_update[D: MutMap](d: D, new: Mapping, /, *, inplace: bool = False) -> D:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References:
        - https://stackoverflow.com/a/30655448
    """
    if not inplace:
        d = deepcopy(d)

    for key, value in new.items():
        match value:
            # recurse on non-empty mapping
            case Mapping() as mapping if mapping:  # non-empty mapping
                subdict = d.get(key, {})
                d[key] = deep_dict_update(subdict, mapping, inplace=True)
            # update value for the given key
            case _:
                d[key] = value
    return d


def nested_paths_exist(paths: Nested[Optional[FilePath]], /) -> bool:
    r"""Check whether the files exist.

    The input can be arbitrarily nested data-structure with `Path` in leaves.
    """
    match paths:
        case None:
            return True
        case str(string):
            return Path(string).exists()
        case Path() as path:
            return path.exists()
        case Mapping() as mapping:
            return all(nested_paths_exist(f) for f in mapping.values())
        case Iterable() as iterable:
            return all(nested_paths_exist(f) for f in iterable)
        case _:
            raise TypeError(f"Unknown type for rawdata_file: {type(paths)}")


def repackage_zip(filepath: FilePath, /) -> None:
    r"""Remove the leading directory from a zip file."""
    original_path = Path(filepath)

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
            logger = logging.getLogger(f"{__name__}/{repackage_zip.__name__}")
            logger.info("Skipping repackage_zip for %s", original_path)
            return

    # create a temporary directory
    with TemporaryDirectory() as temp_dir:
        # move the zip file to the temporary directory
        temp_path = Path(temp_dir) / original_path.name
        shutil.move(original_path, temp_path)
        # create a new zipfile with the modified contents:
        with ZipFile(temp_path) as old_archive, ZipFile(filepath, "w") as new_archive:
            contents = old_archive.namelist()
            top = contents[0]
            for item in tqdm(contents[1:], desc="Repackaging zip file"):
                _, new_name = item.split(top, 1)
                new_archive.writestr(new_name, old_archive.read(item))


def transpose_list_of_dicts[K, V](lst: Iterable[dict[K, V]], /) -> dict[K, list[V]]:
    r"""Fast way to 'transpose' a list of dictionaries.

    Assumptions:
        - all dictionaries have the same keys
        - the keys are always in the same order
        - at least one item in the input
        - can iterate multiple times over lst

    Example:
        >>> list_of_dicts = [
        ...     {"name": "Alice",   "age": 30},
        ...     {"name": "Bob",     "age": 25},
        ...     {"name": "Charlie", "age": 35},
        ... ]  # fmt: skip
        >>> transpose_list_of_dicts(list_of_dicts)
        {'name': ['Alice', 'Bob', 'Charlie'], 'age': [30, 25, 35]}
    """
    keys = next(iter(lst)).keys()
    values = map(list, zip(*(d.values() for d in lst), strict=True))
    return dict(zip(keys, values, strict=True))


def query_bool(question: str, /, *, default: bool) -> bool:
    r"""Ask a yes/no question and returns answer as bool."""
    responses = {"y": True, "yes": True, "n": False, "no": False}
    prompt = "([y]/n)" if default else "(y/[n])"

    for k in range(3):
        msg = (
            f"Invalid response, Please enter either of {responses}" * (k > 0)
            + f"{question} {prompt}:"
        )
        try:
            choice = input(msg).lower()
        except KeyboardInterrupt as exc:
            exc.add_note("Operation aborted.")
            raise

        if not choice and default is not None:
            return default
        if choice in responses:
            return responses[choice]

    raise RuntimeError("Too many invalid responses.")


def query_choice(
    question: str,
    /,
    *,
    choices: set[str],
    default: Optional[str] = None,
    pick_by_number: bool = True,
) -> str:
    r"""Ask the user to pick an option.

    If `pick_by_number=True`, then will allow the user to pick the choice by number.
    """
    choices = set(choices)
    ids: dict[int, str] = dict(enumerate(choices))

    if default is not None and default not in choices:
        raise ValueError(f"Default option {default!r} not in {choices=!r}")

    options = "\n".join(
        f"{k}. {v}" + " (default)" * (v == default) for k, v in enumerate(choices)
    )

    for k in range(3):
        msg = (
            f"{question}\n{options}\nYour choice (int or name):"
            if k == 0
            else f"Please enter either of {choices}"
        )

        try:
            choice = input(msg)
        except KeyboardInterrupt as exc:
            exc.add_note("Operation aborted.")
            raise

        if choice in choices:
            return choice
        if pick_by_number and choice.isdigit() and int(choice) in ids:
            return ids[int(choice)]

    raise RuntimeError("Too many invalid responses.")
