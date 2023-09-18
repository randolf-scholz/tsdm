#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

axes: NDArray[np.object_[plt.Axes]]  # type: ignore[type-var]
fig: plt.Figure
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

for ax in axes.flat:
    # reveal_type(axes)
    ax.set_title("A subplot")
