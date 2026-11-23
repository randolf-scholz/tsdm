r"""Plotting helper functions."""

__all__ = [
    # Functions
    "visualize_distribution",
    "shared_grid_plot",
    "plot_spectrum",
    "center_axes",
    "enable_latex_plotting",
    "rasterize",
    "kernel_heatmap",
]

import shutil
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import colormaps, colors, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText
from matplotlib.typing import RcKeyType
from numpy.typing import ArrayLike, NDArray
from scipy.stats import mode
from torch import Tensor
from torch.linalg import eigvals

from tsdm.constants import EMPTY_MAP

type Location = Literal[
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
    extra_stats: dict | None = None,
) -> None:
    r"""Plot a density histogram of the data on the supplied axes.

    The data is flattened and NaNs are removed. Bin edges are based on the first
    and 99th percentiles, so the histogram may omit values outside that range.
    After removing NaNs, the values must be finite and yield distinct bin edges.
    Logarithmic histograms require positive values.

    Args:
        data: Data to plot.
        ax: Axes to plot into.
        num_bins: Number of bin edges, giving `num_bins - 1` histogram bins.
        log: Whether to use base-10 logarithmic bin edges and logarithmic scales
            on both axes.
        loc: Location of the statistics text box.
        print_stats: Whether to add a table of the NaN fraction, mode, minimum,
            median, maximum, mean, and standard deviation.
        extra_stats: Additional entries for the statistics table. Matching keys
            override the computed entries. Ignored when `print_stats=False`.

    Note:
        In logarithmic mode, statistics other than the NaN fraction describe the
        values retained after filtering by the logarithmic bin range. Otherwise,
        they describe all non-NaN values.
    """
    if isinstance(data, Tensor):
        data = data.detach().cpu().numpy()

    x: NDArray[np.float64] = np.asarray(data, dtype=float).flatten()
    nans = np.isnan(x)
    x = x[~nans]

    ax.grid(axis="x")
    ax.set_axisbelow(True)

    if log:
        base = 10
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

    ax.hist(x, bins=bins.tolist(), density=True)

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
    titles: list[str] | None = None,
    row_headers: list[str] | None = None,
    col_headers: list[str] | None = None,
    xlabels: list[str] | None = None,
    ylabels: list[str] | None = None,
    subplots_kwargs: Mapping[str, Any] = EMPTY_MAP,
    # FIXME: https://github.com/numpy/numpy/issues/24738
) -> tuple[Figure, NDArray]:
    r"""Create a grid of plots with shared axes and optional row and column headers.

    Args:
        data: Array whose first two dimensions select the subplot row and column.
            Two-dimensional input is treated as a single row of plots, with one
            plot per original row of data.
        plot_func: Function called as `plot_func(data_slice, ax=ax, **plot_kwargs)`
            for each subplot.
        plot_kwargs: Additional keyword arguments for `plot_func`.
        titles: Subplot titles in row-major order, with one title per subplot.
        row_headers: One header per subplot row.
        col_headers: One header per subplot column.
        xlabels: One x-axis label per column, applied to the bottom row.
        ylabels: One y-axis label per row, applied to the leftmost column.
        subplots_kwargs: Options for `matplotlib.pyplot.subplots`. By default,
            x-axes are shared within columns and y-axes within rows.

    Returns:
        The figure and a two-dimensional array of subplot axes.

    References:
        - [Row and column headers for Matplotlib subplots](https://stackoverflow.com/a/25814386).
    """
    array = np.array(data)

    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)

    nrows, ncols = array.shape[:2]

    opts: dict[str, Any] = {
        "figsize": (5 * ncols, 3 * nrows),
        "sharex": "col",
        "sharey": "row",
        "squeeze": False,
        "tight_layout": True,
    } | dict(subplots_kwargs)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, **opts)

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
    r"""Plot a square matrix's eigenvalues in the complex plane.

    Args:
        kernel: Two-dimensional square tensor or NumPy array. NumPy arrays are
            converted to tensors with dtype `torch.float32`.
        style: Matplotlib style to apply while creating the plot.
        axis_kwargs: Keyword arguments for `Axes.set`.
        figure_kwargs: Keyword arguments for `matplotlib.pyplot.subplots`.
        scatter_kwargs: Keyword arguments for `Axes.scatter`.

    Returns:
        Figure containing the eigenvalue scatter plot.

    Raises:
        ValueError: If `kernel` is not a two-dimensional square matrix.
    """
    axis_opts: dict[str, Any] = {
        "xlim": (-2.5, +2.5),
        "ylim": (-2.5, +2.5),
        "aspect": "equal",
        "ylabel": "imag part",
        "xlabel": "real part",
    } | dict(axis_kwargs)

    # default: 1024px×1024px
    fig_opts: dict[str, Any] = {
        "figsize": (4, 4),
        "constrained_layout": True,
        "dpi": 256,
    } | dict(figure_kwargs)

    scatter_opts: dict[str, Any] = {
        "edgecolors": "none",
    } | dict(scatter_kwargs)

    if not isinstance(kernel, Tensor):
        kernel = torch.tensor(kernel, dtype=torch.float32)

    # validate kernel shape
    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D kernel, got {kernel.ndim=}")
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Expected square kernel, got {kernel.shape=}")

    with plt.style.context(style):
        eigs = eigvals(kernel).detach().cpu()
        fig, ax = plt.subplots(**fig_opts)
        ax.set(**axis_opts)
        ax.scatter(eigs.real, eigs.imag, **scatter_opts)

    return fig


def center_axes(fig: Figure, /, *, remove_labels: bool = True) -> Figure:
    r"""Move the left and bottom spines of every subplot to the data origin.

    Hide the top and right spines and add arrowhead markers to the visible axes.

    Args:
        fig: Figure to modify in place.
        remove_labels: Whether to clear the x-axis and y-axis labels.

    Returns:
        The modified input figure.
    """
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


def enable_latex_plotting() -> None:
    r"""Configure Matplotlib to render with LaTeX using the PGF backend.

    If `lualatex` is available, apply `MATPLOTLIB_CONFIG` and select the PGF
    backend globally. Emit a warning if `lualatex` is missing or configuration
    raises `ValueError`.
    """
    PGF_PREAMBLE = r"""
    \usepackage{fontspec}
    \usepackage[T1]{fontenc}
    \usepackage[utf8x]{inputenc}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    \usepackage{unicode-math}
    """

    LATEX_PREAMBLE = r"""
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    """

    MATPLOTLIB_CONFIG: dict[RcKeyType, Any] = {
        # "mathtext.fontset": "stix",
        # "font.family": "STIXGeneral",
        # "svg.fonttype": "none",
        "text.usetex": True,
        "pgf.texsystem": r"lualatex",
        "pgf.preamble": PGF_PREAMBLE,
        "text.latex.preamble": LATEX_PREAMBLE,
    }

    try:
        if shutil.which("lualatex") is not None:
            mpl.rcParams.update(MATPLOTLIB_CONFIG)
            mpl.use("pgf")
        else:
            warnings.warn(
                "lualatex not found. Using default matplotlib backend.", stacklevel=1
            )
    except ValueError:
        warnings.warn(
            "matplotlib: pgf backend not available / no LaTeX rendering!", stacklevel=1
        )


@torch.no_grad()
def kernel_heatmap(
    kernel: NDArray | Tensor,  # Float[..., $H, $W]
    /,
    *,
    fmt: Literal["HWC", "CHW"] = "HWC",
    cmap: str | colors.Colormap = "seismic",
) -> NDArray:  # Float[..., $H, $W, 3] if "HWC" or Float[..., 3, $H, $W] if "CHW"
    r"""Map a matrix to an RGB heatmap.

    Values are centered at ½, scaled by six times the input's standard deviation,
    and clipped to $[0, 1]$ before applying the colormap. For normally distributed
    input, approximately 99.7% of values lie in this interval before clipping.
    This affine transformation does not make arbitrary input normally distributed.

    Args:
        kernel: Matrix of scalar values. Values must be finite and have nonzero
            standard deviation for the normalization to be well-defined.
        fmt: Output layout. `"HWC"` appends the RGB channel axis; `"CHW"` places
            it first. This option does not change how the input is interpreted.
        cmap: Registered colormap name or a Matplotlib colormap.

    Returns:
        RGB array with values in $[0, 1]$. For input of shape `(H, W)`, the
        output has shape `(H, W, 3)` or `(3, H, W)`, according to `fmt`.

    Raises:
        ValueError: If `fmt` is neither `"HWC"` nor `"CHW"`.
    """
    # This transformation is chosen because by the 68–95–99.7 rule,
    # for k=6=2⋅3 roughly 99.7% of the probability mass will lie in the interval [0, 1]
    kernel = 0.5 + (kernel - kernel.mean()) / (6 * kernel.std())
    kernel = kernel.clip(0, 1)

    if isinstance(kernel, Tensor):
        kernel = kernel.cpu().numpy()

    colormap = colormaps[cmap] if isinstance(cmap, str) else cmap
    rgba: NDArray = colormap(kernel)
    rgb = rgba[..., :-1]

    match fmt:
        case "HWC":
            return rgb
        case "CHW":
            return np.rollaxis(rgb, -1)
        case _:
            raise ValueError(f"Invalid format {fmt!r}")


def rasterize(
    fig: Figure,
    /,
    *,
    w: int = 3,
    h: int = 3,
    px: int = 512,
    py: int = 512,
) -> np.ndarray:
    r"""Render a figure as an RGBA image using the Agg backend.

    The figure's size and DPI are changed in place. The DPI is the floored
    arithmetic mean of `px / w` and `py / h`. Choose dimensions whose Agg
    render size agrees with `(px, py)`; arbitrary combinations of physical
    and pixel dimensions are not guaranteed to work.

    Args:
        fig: Figure to rasterize.
        w: Positive width of the figure in inches.
        h: Positive height of the figure in inches.
        px: Requested width of the image in pixels.
        py: Requested height of the image in pixels.

    Returns:
        Array of shape `(py, px, 4)` with `uint8` RGBA values.

    Raises:
        ValueError: If the rendered buffer cannot be reshaped to `(py, px, 4)`.
    """
    dpi = (px / w + py / h) // 2  # compromise
    fig.set_dpi(dpi)
    fig.set_size_inches(w, h, forward=True)

    # Render via Agg and extract RGBA buffer without Pillow.
    fig.canvas.draw()
    rgba_buf = FigureCanvasAgg(fig).buffer_rgba()
    rgba = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((py, px, 4))

    return rgba
