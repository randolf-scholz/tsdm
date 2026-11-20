r"""Utilities shared by backend implementations."""

__all__ = ["get_outlier_options"]

from collections.abc import Mapping
from typing import Any


def get_outlier_options(
    limits: Mapping[str, Any] | None = None,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = None,
    upper_bound: Mapping[Any, float | None] | float | None = None,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = None,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = None,
) -> dict[str, Any]:
    r"""Validate and normalize outlier bounds from mappings or keyword arguments.

    With `limits`, every boundary keyword must be `None`. Otherwise, missing
    or `None` bounds leave that side unbounded. Explicit `None` keywords may
    accompany `limits`; they are equivalent to omitted keywords.

    Mappings may be organized by bound name or variable name. Native limits
    tables must first be converted to mappings by their backend. Each bounded
    column must have entries in all four boundary mappings.
    """
    lims: dict[str, Any] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }
    if limits is None:
        return lims
    if any(value is not None for value in lims.values()):
        raise ValueError("Limits specified both as positional and keyword arguments.")
    if not isinstance(limits, Mapping):
        raise TypeError(f"Expected a mapping of outlier limits, got {type(limits)}.")
    if lims.keys() <= limits.keys():
        return {key: limits[key] for key in lims}
    return {
        key: {variable: bounds[key] for variable, bounds in limits.items()}
        for key in lims
    }
