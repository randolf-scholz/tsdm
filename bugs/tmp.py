#!/usr/bin/env python3

from collections.abc import Mapping
from typing import Any


def make_plot(subplot_kwargs: Mapping[str, Any]) -> None:
    subplot_kwargs = {
        "option1": "default1",
        "option2": "default2",
    } | subplot_kwargs
    # plotfn(..., **subplot_kwargs)
