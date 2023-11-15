#!/usr/bin/env python3

from typing import Any, cast

x: float = 1.0

x = cast(Any, x)

reveal_type(x)

x = cast(str, x)

reveal_type(x)
