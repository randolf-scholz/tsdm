r"""Utility functions."""

__all__ = [
    # Classes
    # Functions
    "flatten_dict",
    "unflatten_dict",
    "nested_paths_exist",
    "transpose_list_of_dicts",
    "prompt_choice",
    "prompt_yes_no",
]

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, cast, overload

from tsdm.types.aliases import FilePath, Nested, NestedDict, NestedMapping


@overload
def flatten_dict(  # pyrefly: ignore[inconsistent-overload-default]
    d: NestedMapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> dict[str, Any]: ...
@overload
def flatten_dict[Key, Key_flat](
    d: NestedMapping[Key, Any],
    /,
    *,
    join_fn: Callable[[Iterable[Key]], Key_flat],
    split_fn: Callable[[Key_flat], Iterable[Key]],
    recursive: bool | int = ...,
) -> dict[Key_flat, Any]: ...
def flatten_dict[Key, Key_flat](
    d: NestedMapping[Key, Any],
    /,
    *,
    join_fn: Callable[[Iterable[Key]], Key_flat] = cast("Any", ".".join),  # ruff: ignore[B008]
    split_fn: Callable[[Key_flat], Iterable[Key]] = cast("Any", lambda s: s.split(".")),  # ruff: ignore[B008]
    recursive: bool | int = True,
) -> dict[Key_flat, Any]:
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
    result: dict[Key_flat, Any] = {}
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
def unflatten_dict(  # pyrefly: ignore[inconsistent-overload-default]
    d: Mapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> NestedDict[str, Any]: ...
@overload
def unflatten_dict[Key, Key_flat](
    d: Mapping[Key_flat, Any],
    /,
    *,
    join_fn: Callable[[Iterable[Key]], Key_flat],
    split_fn: Callable[[Key_flat], Iterable[Key]],
    recursive: bool | int = ...,
) -> NestedDict[Key, Any]: ...
def unflatten_dict[Key, Key_flat](
    d: Mapping[Key_flat, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[Key]], Key_flat] = cast("Any", ".".join),  # ruff: ignore[B008]
    split_fn: Callable[[Key_flat], Iterable[Key]] = cast("Any", lambda s: s.split(".")),  # ruff: ignore[B008]
) -> NestedDict[Key, Any]:
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
    result: dict[Key, Any] = {}
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


def prompt_yes_no(question: str, /, *, default: bool) -> bool:
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


def prompt_choice(
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
