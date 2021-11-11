r"""Plotting helper functions."""

from __future__ import annotations

__all__ = [
    # Functions
    "visualize_distribution",
    "shared_grid_plot",
    "plot_spectrum",
]

import logging
from typing import Callable, Literal, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.pyplot import Axes, Figure
from numpy.typing import ArrayLike
from scipy.stats import mode
from torch import Tensor
from torch.linalg import eigvals


__logger__ = logging.getLogger(__name__)

Location = Literal[
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
    x: ArrayLike,
    ax: Axes,
    num_bins: int = 50,
    log: bool = True,
    loc: Location = "upper right",
    print_stats: bool = True,
    extra_stats: Optional[dict[str, str]] = None,
):
    r"""Plot the distribution of x in the given axis.

    Parameters
    ----------
    x: ArrayLike
    ax: Axes
    num_bins: int or Sequence[int]
    log: bool or float, default=False
        if True, use log base 10, if float, use  log w.r.t. this base
    loc: Location
    print_stats: bool
    extra_stats: Optional[dict[str, str]]
        Additional things to add to the stats table
    """
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    x = np.asanyarray(x)

    x = x.flatten().astype(float)
    nans = np.isnan(x)
    x = x[~nans]

    ax.grid(axis="x")
    ax.set_axisbelow(True)

    if log:
        base = 10 if log is True else log
        tol = 2 ** -24 if np.issubdtype(x.dtype, np.float32) else 2 ** -53  # type: ignore
        z = np.log10(np.maximum(x, tol))
        ax.set_xscale("log", base=base)
        ax.set_yscale("log", base=base)
        low = np.floor(np.quantile(z, 0.01))
        high = np.ceil(np.quantile(z, 1 - 0.01))
        x = x[(z >= low) & (z <= high)]  # type: ignore
        bins = np.logspace(low, high, num=num_bins, base=10)
    else:
        low = np.quantile(x, 0.01)
        high = np.quantile(x, 1 - 0.01)
        bins = np.linspace(low, high, num=num_bins)

    ax.hist(x, bins=bins, density=True)

    if print_stats:
        stats = {
            "NaNs": f"{100 * np.mean(nans):.2f}" + r"\%",
            "Mean": f"{np.mean(x):.2e}",
            "Median": f"{np.median(x):.2e}",
            "Mode": f"{mode(x)[0][0]:.2e}",
            "Stdev": f"{np.std(x):.2e}",
        }
        if extra_stats is not None:
            stats |= {str(key): str(val) for key, val in extra_stats.items()}

        pad = max(len(key) for key in stats)

        table = (
            r"\scriptsize"
            + r"\begin{tabular}{ll}"
            + r"\\ ".join([key.ljust(pad) + " & " + val for key, val in stats.items()])
            + r"\end{tabular}"
        )

        # if extra_stats is not None:
        __logger__.info("writing table %s", table)

        # text = r"\begin{tabular}{ll}test & and\\ more &test\end{tabular}"
        textbox = AnchoredText(table, loc=loc, borderpad=0.0)
        ax.add_artist(textbox)


@torch.no_grad()
def shared_grid_plot(
    data: ArrayLike,
    plot_func: Callable[..., None],
    plot_kwargs: dict = None,
    titles: list[str] = None,
    row_headers: list[str] = None,
    col_headers: list[str] = None,
    xlabels: list[str] = None,
    ylabels: list[str] = None,
    **subplots_kwargs: dict,
) -> tuple[Figure, Axes]:
    r"""Create a grid plot with shared axes and row/col headers.

    Based on https://stackoverflow.com/a/25814386/9318372

    Parameters
    ----------
    data: ArrayLike
    plot_func: Callable
        With signature plot_func(data, ax=)
    plot_kwargs: dict
    titles: list[str]
    row_headers: list[str]
    col_headers: list[str]
    xlabels: list[str]
    ylabels: list[str]
    subplots_kwargs: dict
        Default arguments: ``tight_layout=True``, ``sharex='col``, ``sharey='row'``

    Returns
    -------
    Figure
    Axes
    """
    data = np.array(data)

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    nrows, ncols = data.shape[:2]  # type: ignore

    _subplot_kwargs = {
        "figsize": (5 * ncols, 3 * nrows),
        "sharex": "col",
        "sharey": "row",
        "squeeze": False,
        "tight_layout": True,
    }

    if subplots_kwargs is not None:
        _subplot_kwargs.update(subplots_kwargs)  # type: ignore

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **_subplot_kwargs)

    # call the plot functions
    for idx in np.ndindex(axes.shape):
        plot_func(data[idx], ax=axes[idx], **plot_kwargs)  # type: ignore

    # set axes titles
    if titles is not None:
        for ax, title in np.nditer([axes, titles]):  # type: ignore
            ax.set_title(title)

    # set axes x-labels
    if xlabels is not None:
        for ax, xlabel in np.nditer([axes[-1], xlabels], flags=["refs_ok"]):  # type: ignore
            ax.item().set_xlabel(xlabel)

    # set axes y-labels
    if ylabels is not None:
        for ax, ylabel in np.nditer([axes[:, 0], ylabels], flags=["refs_ok"]):  # type: ignore
            ax.item().set_ylabel(ylabel)

    pad = 5  # in points

    # set axes col headers
    if col_headers is not None:
        for ax, col_header in zip(axes[0], col_headers):
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
        for ax, row_header in zip(axes[:, 0], row_headers):
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
    kernel: Tensor,
    style: str = "ggplot",
    axis_kwargs: Optional[dict] = None,
    figure_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
) -> Figure:
    r"""Create scatter-plot of complex matrix eigenvalues.

    Parameters
    ----------
    kernel: Tensor
    style: str = "bmh"
        Which matplotlib style to use.
    axis_kwargs: Optional[dict] = None,
        Keyword-Arguments to pass to ``Axes.set``
    figure_kwargs: Optional[dict] = None
        Keyword-Arguments to pass to ``matplotlib.pyplot.subplots``
    scatter_kwargs: Optional[dict] = None
        Keyword-Arguments to pass to ``matplotlib.pyplot.scatter``

    Returns
    -------
    Figure
    """
    axis_kwargs = {
        "xlim": (-2.5, +2.5),
        "ylim": (-2.5, +2.5),
        "aspect": "equal",
        "ylabel": "imag part",
        "xlabel": "real part",
    } | (axis_kwargs or {})

    figure_kwargs = {
        "figsize": (4, 4),
        "constrained_layout": True,
        "dpi": 256,  # default: 1024px×1024px
    } | (figure_kwargs or {})

    scatter_kwargs = {
        "edgecolors": "none",
    } | (scatter_kwargs or {})

    with plt.style.context(style):
        assert len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1]
        eigs = eigvals(kernel).detach().cpu()
        fig, ax = plt.subplots(**figure_kwargs)
        ax.set(**axis_kwargs)
        ax.scatter(eigs.real, eigs.imag, **scatter_kwargs)
        return fig


# @torch.no_grad()
# def plot_kernel_heatmap(kernel: Tensor, cmap: str = "seismic"):
#     kernel = kernel.clone().detach().cpu()
#     assert len(kernel.shape)==2 and kernel.shape[0] == kernel.shape[1]
#     cmap = cm.get_cmap("seismic")
#     RGBA = cmap(kernel)
#     return RGBA[..., :-1]
