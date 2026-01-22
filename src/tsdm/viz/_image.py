r"""Visualization Utilities for image data."""

__all__ = [
    # Functions
    "kernel_heatmap",
    "rasterize",
]

from typing import Literal

import numpy as np
import torch
from matplotlib import colormaps, colors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor


@torch.no_grad()
def kernel_heatmap(
    kernel: NDArray | Tensor,
    /,
    *,
    fmt: Literal["HWC", "CHW"] = "HWC",
    cmap: str | colors.Colormap = "seismic",
) -> NDArray:
    r"""Create heatmap of given matrix.

    .. signature:: ``(..., ) ⟶ (..., 3)`` if "HWC"
    .. signature:: ``(..., ) ⟶ (3, ...)`` if "CHW".

    By default, the data is linearly transformed to a normal distribution $𝓝(½,⅙)$,
    which ensures that 99.7% of the data lies in the interval $[0,1]$, and then clipped.

    Use `fmt` to specify whether the input is height×width×channels or channels×height×width.
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
    r"""Convert a figure to image with specific pixel size.

    The dpi setting will be automatically determined as the average of the
    horizontal and vertical dpi.

    Args:
        fig: Figure to rasterize.
        w: Width of the figure in inches.
        h: Height of the figure in inches.
        px: Width of the figure in pixels.
        py: Height of the figure in pixels.
    """
    dpi = (px / w + py / h) // 2  # compromise
    fig.set_dpi(dpi)
    fig.set_size_inches(w, h, forward=True)

    # Render via Agg and extract RGBA buffer without Pillow.
    fig.canvas.draw()
    rgba_buf = FigureCanvasAgg(fig).buffer_rgba()
    rgba = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((py, px, 4))

    return rgba
