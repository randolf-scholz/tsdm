import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

axes: NDArray[Axes]  # ✘ mypy: raises type-var
fig: Figure
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

for ax in axes.flat:
    reveal_type(axes)  # Axes if the previous error is ignored, else ndarray
    ax.set_title("A subplot")
