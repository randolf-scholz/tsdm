r"""Visualization Utilities for image data."""

__all__ = [
    # Functions
    "kernel_heatmap",
    "rasterize",
]

from tempfile import TemporaryFile
from typing import Literal

import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor


@torch.no_grad()
def kernel_heatmap(
    kernel: NDArray | Tensor,
    /,
    *,
    fmt: Literal["HWC", "CHW"] = "HWC",
    cmap: colors.Colormap = "seismic",
) -> NDArray:
    r"""Create heatmap of given matrix.

    .. Signature:: ``(..., ) ⟶ (..., 3)`` if "HWC" or ``(..., ) ⟶ (3, ...)`` if "CHW".

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

    colormap = cm.get_cmap(cmap)
    RGBA = colormap(kernel)
    RGB = RGBA[..., :-1]

    if fmt == "HWC":
        return RGB
    if fmt == "CHW":
        return np.rollaxis(RGB, -1)
    raise ValueError(fmt)


def rasterize(
    fig: Figure,
    /,
    *,
    w: int = 3,
    h: int = 3,
    px: int = 512,
    py: int = 512,
) -> np.ndarray:
    r"""Convert figure to image with specific pixel size.

    The dpi setting will be automatically determined as the average of the
    horizontal and vertical dpi.

    Args:
        fig: Figure to rasterize.
        w: Width of figure in inches.
        h: Height of figure in inches.
        px: Width of figure in pixels.
        py: Height of figure in pixels.
    """
    dpi = (px / w + py / h) // 2  # compromise
    fig.set_dpi(dpi)
    fig.set_size_inches(w, h)

    # we serialize with PIL and return the array
    with TemporaryFile(suffix=".png") as file:
        fig.savefig(file, dpi=dpi)
        im = Image.open(file)
        arr = np.array(im)

    return arr
