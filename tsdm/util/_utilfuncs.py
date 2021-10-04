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
    "now",
]


import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Iterable, NamedTuple

from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


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
