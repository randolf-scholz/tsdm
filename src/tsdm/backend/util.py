r"""Utilities shared by backend implementations."""

__all__ = ["get_outlier_options"]

from collections.abc import Mapping, Sequence
from typing import Any

from tsdm.constants import UNDEFINED


def _extract_limits(
    limits: Mapping[str, Any], keys: Sequence[str], /
) -> dict[str, Any]:
    r"""Normalize mappings organized by bound name or variable name."""
    if set(keys).issubset(limits):
        return {key: limits[key] for key in keys}
    return {
        key: {variable: bounds[key] for variable, bounds in limits.items()}
        for key in keys
    }


def get_outlier_options(
    limits: Mapping[str, Any],
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None,
    upper_bound: Mapping[Any, float | None] | float | None,
    lower_inclusive: Mapping[Any, bool | None] | bool | None,
    upper_inclusive: Mapping[Any, bool | None] | bool | None,
) -> Mapping[str, Any]:
    r"""Validate and normalize outlier bounds from mappings or keyword arguments.

    Supply either a mapping of limits or all four keyword bounds, using
    `UNDEFINED` for omitted arguments. Mappings may be organized by bound name
    or variable name. Native limits tables must first be converted to mappings
    by their backend.
    """
    lims: Mapping[str, Any] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }
    undefined = [value is UNDEFINED for value in lims.values()]

    if limits is UNDEFINED:
        if any(undefined):
            raise ValueError(f"Missing boundary values: {lims}")
        return lims
    if all(undefined):
        if not isinstance(limits, Mapping):
            raise TypeError(
                f"Expected a mapping of outlier limits, got {type(limits)}."
            )
        return _extract_limits(limits, list(lims))
    raise ValueError("Limits specified both as positional and keyword arguments.")
