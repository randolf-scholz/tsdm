#!/usr/bin/env python

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gmean(x: ArrayLike) -> np.floating:
    r"""Geometric mean."""
    z = np.asarray(x)
    return np.prod(np.abs(z)) ** (1 / len(z))


reveal_type(np.float32(3.2) ** np.arange(3))
