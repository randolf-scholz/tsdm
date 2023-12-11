#!/usr/bin/env python


import numpy as np
from numpy.typing import NDArray

a: NDArray[np.floating] = np.random.randn(10)
b: NDArray[np.bool_] = a > 0.5
