r"""Plotting helper functions."""

__all__ = [
    # Functions
    "visualize_distribution",
    "shared_grid_plot",
    "plot_spectrum",
    "center_axes",
]

from collections.abc import Callable, Mapping

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike, NDArray
from scipy.stats import mode
from torch import Tensor
from torch.linalg import eigvals
from typing_extensions import Any, Literal, Optional, TypeAlias

from tsdm.constants import EMPTY_MAP

Location: TypeAlias = Literal[
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


@torch.no_grad()
def visualize_distribution(
    data: ArrayLike,
    /,
    *,
    ax: Axes,
    num_bins: int = 50,
    log: bool = True,
    loc: Location = "upper right",
    print_stats: bool = True,
    extra_stats: Optional[dict] = None,
) -> None:
    r"""Plot the distribution of x in the given figure axes.

    Args:
        data: Data to plot.
        ax: Axes to plot into.
        num_bins: Number of bins to use for histogram.
        log: If True, use log base 10, if `float`, use  log w.r.t. this base
        loc: Location of 'stats' text.
        print_stats: Add table of mean, std, min, max, median, mode to plot
        extra_stats: Additional things to add to the 'stats' table
    """
    if isinstance(data, Tensor):
        data = data.detach().cpu().numpy()

    x: NDArray[np.float64] = np.asarray(data, dtype=float).flatten()
    nans = np.isnan(x)
    x = x[~nans]

    ax.grid(axis="x")
    ax.set_axisbelow(True)

    if log:
        base = 10 if log is True else log
        tol = 2**-24 if np.issubdtype(x.dtype, np.float32) else 2**-53
        z = np.log10(np.maximum(x, tol))
        ax.set_xscale("log", base=base)
        ax.set_yscale("log", base=base)
        low = np.floor(np.quantile(z, 0.01))
        high = np.ceil(np.quantile(z, 1 - 0.01))
        x = x[(z >= low) & (z <= high)]
        bins = np.logspace(low, high, num=num_bins, base=10)
    else:
        low = np.quantile(x, 0.01)
        high = np.quantile(x, 1 - 0.01)
        bins = np.linspace(low, high, num=num_bins)

    ax.hist(x, bins=bins, density=True)  # type: ignore[arg-type]

    if print_stats:
        stats = {
            "NaNs": f"{100 * np.mean(nans):6.2%}",
            "Mode": f"{mode(x)[0]: .2g}",
            "Min": f"{np.min(x): .2g}",
            "Median": f"{np.median(x): .2g}",
            "Max": f"{np.max(x): .2g}",
            "Mean": f"{np.mean(x): .2g}",
            "Stdev": f"{np.std(x): .2g}",
        }
        if extra_stats is not None:
            stats |= {str(key): str(val) for key, val in extra_stats.items()}

        pad = max(map(len, stats), default=0)
        table = "\n".join([f"{key:<{pad}}  {val}" for key, val in stats.items()])

        # use mono-spaced font
        textbox = AnchoredText(
            table,
            loc=loc,
            borderpad=0.0,
            prop={"family": "monospace"},
        )
        textbox.patch.set_alpha(0.8)
        ax.add_artist(textbox)


@torch.no_grad()
def shared_grid_plot(
    data: ArrayLike,
    /,
    *,
    plot_func: Callable[..., None],
    plot_kwargs: Mapping[str, Any] = EMPTY_MAP,
    titles: Optional[list[str]] = None,
    row_headers: Optional[list[str]] = None,
    col_headers: Optional[list[str]] = None,
    xlabels: Optional[list[str]] = None,
    ylabels: Optional[list[str]] = None,
    subplots_kwargs: Mapping[str, Any] = EMPTY_MAP,
) -> tuple[Figure, np.ndarray[Axes]]:  # type: ignore[type-arg]
    r"""Create a compute_grid plot with shared axes and row/col headers.

    References:
        - https://stackoverflow.com/a/25814386
    """
    array = np.array(data)

    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)

    nrows, ncols = array.shape[:2]

    subplots_kwargs = {
        "figsize": (5 * ncols, 3 * nrows),
        "sharex": "col",
        "sharey": "row",
        "squeeze": False,
        "tight_layout": True,
    } | dict(subplots_kwargs)

    axes: NDArray[Axes]  # type: ignore[type-var]
    fig: Figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subplots_kwargs)  # type: ignore[arg-type]

    # call the plot functions
    for idx in np.ndindex(axes.shape):
        plot_func(array[idx], ax=axes[idx], **plot_kwargs)

    # set axes titles
    if titles is not None:
        # for ax, title in np.nditer([axes, titles]):
        for ax, title in zip(axes.flat, np.asarray(titles).flat, strict=True):
            ax.set_title(title)

    # set axes x-labels
    if xlabels is not None:
        # for ax, xlabel in np.nditer([axes[-1], xlabels], flags=["refs_ok"]):
        for ax, xlabel in zip(axes[-1], np.asarray(xlabels).flat, strict=True):
            ax.set_xlabel(xlabel)

    # set axes y-labels
    if ylabels is not None:
        # for ax, ylabel in np.nditer([axes[:, 0], ylabels], flags=["refs_ok"]):
        for ax, ylabel in zip(axes[:, 0], np.asarray(ylabels).flat, strict=True):
            ax.set_ylabel(ylabel)

    pad = 5  # in points

    # set axes col headers
    if col_headers is not None:
        for ax, col_header in zip(axes[0], col_headers, strict=True):
            ax.annotate(
                col_header,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size="large",
                ha="center",
                va="baseline",
            )

    # set axes row headers
    if row_headers is not None:
        for ax, row_header in zip(axes[:, 0], row_headers, strict=True):
            ax.annotate(
                row_header,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )

    return fig, axes


@torch.no_grad()
def plot_spectrum(
    kernel: Tensor | NDArray,
    /,
    *,
    style: str = "ggplot",
    axis_kwargs: Mapping[str, Any] = EMPTY_MAP,
    figure_kwargs: Mapping[str, Any] = EMPTY_MAP,
    scatter_kwargs: Mapping[str, Any] = EMPTY_MAP,
) -> Figure:
    r"""Create a scatter-plot of complex matrix eigenvalues.

    Arguments:
        kernel: Tensor
        style: Which matplotlib style to use.
        axis_kwargs: Keyword-Arguments to pass to `Axes.set`
        figure_kwargs: Keyword-Arguments to pass to `matplotlib.pyplot.subplots`
        scatter_kwargs: Keyword-Arguments to pass to `matplotlib.pyplot.scatter`
    """
    axis_kwargs = {
        "xlim": (-2.5, +2.5),
        "ylim": (-2.5, +2.5),
        "aspect": "equal",
        "ylabel": "imag part",
        "xlabel": "real part",
    } | dict(axis_kwargs)

    figure_kwargs = {
        "figsize": (4, 4),
        "constrained_layout": True,
        "dpi": 256,  # default: 1024px×1024px
    } | dict(figure_kwargs)

    scatter_kwargs = {
        "edgecolors": "none",
    } | dict(scatter_kwargs)

    if not isinstance(kernel, Tensor):
        kernel = torch.tensor(kernel, dtype=torch.float32)

    with plt.style.context(style):
        assert len(kernel.shape) == 2
        assert kernel.shape[0] == kernel.shape[1]
        eigs = eigvals(kernel).detach().cpu()
        fig, ax = plt.subplots(**figure_kwargs)  # type: ignore[arg-type]
        ax.set(**axis_kwargs)
        ax.scatter(eigs.real, eigs.imag, **scatter_kwargs)

    return fig


def center_axes(fig: Figure, /, *, remove_labels: bool = True) -> Figure:
    r"""Center axes in the figure."""
    for ax in fig.axes:
        if remove_labels:
            ax.set(xlabel="", ylabel="")
        ax.spines["left"].set_position(("data", 0))
        ax.spines["left"].set_color("k")
        ax.spines["bottom"].set_color("k")
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.plot(0.99, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=True)
        ax.plot(0, 0.99, "^k", transform=ax.get_xaxis_transform(), clip_on=True)
    return fig
