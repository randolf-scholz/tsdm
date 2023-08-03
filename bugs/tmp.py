#!/usr/bin/env python3

from typing import Any, Mapping


def joint_keys(*dicts: Mapping[str, Any]) -> set[str]:
    """Find joint keys in collection of dictionaries."""
    return set.intersection(*map(set, dicts))


def joint_keys2(*dicts: Mapping[str, Any]) -> set[str]:
    """Find joint keys in collection of dictionaries."""
    dicts_keys: map[set[str]] = map(set, dicts)
    return set.intersection(*dicts_keys)
