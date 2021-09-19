r"""Plotting helper functions."""

import logging
from typing import Callable, Final, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.pyplot import Axes, Figure
from numpy.typing import ArrayLike
from scipy.stats import mode
from torch import Tensor

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["visualize_distribution", "shared_grid_plot"]


def visualize_distribution(
    x,
    ax,
    bins: int = 50,
    log: bool = True,
    loc: int = 1,
    print_stats: bool = True,
    extra_stats: Optional[dict[str, str]] = None,
):
    r"""Plot the distribution of x in the given axis.

    Parameters
    ----------
    x: ArrayLike
    ax: Axes
    bins: int or Sequence[int]
    log: bool or float, default=False
        if True, use log base 10, if float, use  log w.r.t. this base
    loc: int or str
    print_stats: bool
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
        bins = np.logspace(low, high, num=bins, base=10)

    ax.hist(x, bins=bins, density=True)

    if print_stats:
        stats = {
            "NaNs": f"{100*np.mean(nans):.2f}" + r"\%",
            "Mean": f"{np.mean(x):.2e}",
            "Median": f"{np.median(x):.2e}",
            "Mode": f"{mode(x)[0][0]:.2e}",
            "Stdev": f"{np.std(x):.2e}",
        }
        if extra_stats is not None:
            stats |= {str(key): str(val) for key, val in extra_stats.items()}

        pad = max(len(key) for key in stats)

        table = (
            r"\begin{tabular}{ll}"
            + r"\\ ".join([key.ljust(pad) + " & " + val for key, val in stats.items()])
            + r"\end{tabular}"
        )

        # if extra_stats is not None:
        logger.info("writing table %s", table)

        # text = r"\begin{tabular}{ll}test & and\\ more &test\end{tabular}"
        textbox = AnchoredText(table, loc=loc, borderpad=0)
        ax.add_artist(textbox)


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

    NROWS, NCOLS = data.shape[:2]  # type: ignore

    SUBPLOT_KWARGS = {
        "figsize": (5 * NCOLS, 3 * NROWS),
        "sharex": "col",
        "sharey": "row",
        "squeeze": False,
        "tight_layout": True,
    }

    if subplots_kwargs is not None:
        SUBPLOT_KWARGS.update(subplots_kwargs)

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, **SUBPLOT_KWARGS)

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
