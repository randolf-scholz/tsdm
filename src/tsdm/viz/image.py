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
