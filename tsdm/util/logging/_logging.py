r"""Module Summary Line.

Module description
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

import logging
from typing import Final

import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import Figure
from torch import Tensor, jit, trace
from torch.linalg import cond, det, eigvals, matrix_norm, matrix_rank, slogdet
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.optimizers import Optimizer
from tsdm.util._norms import multi_norm

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "log_kernel_information",
    "log_optimizer_state",
]


def plot_spectrum(kernel: Tensor) -> Figure:
    r"""Create scatter-plot of complex matrix eigenvalues.

    Parameters
    ----------
    kernel: Tensor

    Returns
    -------
    Figure
    """
    assert len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1]
    eigs = eigvals(kernel).detach().cpu()
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    ax.set_xlim([-2.5, +2.5])
    ax.set_ylim([-2.5, +2.5])
    ax.set_aspect("equal")
    ax.set_xlabel("real part")
    ax.set_ylabel("imag part")
    ax.scatter(eigs.real, eigs.imag)
    return fig


@jit.script
def symmetric(x: Tensor) -> Tensor:
    r"""Symmetric part of matrix."""
    return (x + x.swapaxes(-1, -2)) / 2


@jit.script
def skew_symmetric(x: Tensor) -> Tensor:
    r"""Skew-Symmetric part of matrix."""
    return (x - x.swapaxes(-1, -2)) / 2


@jit.script
def symmpart(kernel):
    r"""Relative magnitude of symmetric part."""
    return torch.mean(symmetric(kernel) ** 2) / torch.mean(kernel ** 2)


@jit.script
def skewpart(kernel):
    r"""Relative magnitude of skew-symmetric part."""
    return torch.mean(skew_symmetric(kernel) ** 2) / torch.mean(kernel ** 2)


def log_kernel_information(
    i: int, writer: SummaryWriter, kernel: Tensor, histograms: bool = False
):
    """Log kernel information.

    Parameters
    ----------
    i: int
    writer: SummaryWriter
    kernel: Tensor
    histograms: bool = False
    """
    assert len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1]
    writer.add_scalar("kernel/skewpart", skewpart(kernel), i)
    writer.add_scalar("kernel/symmpart", symmpart(kernel), i)

    # general properties
    writer.add_scalar("kernel/det", det(kernel), i)
    writer.add_scalar("kernel/rank", matrix_rank(kernel), i)
    writer.add_scalar("kernel/trace", trace(kernel), i)
    writer.add_scalar("kernel/cond", cond(kernel), i)
    writer.add_scalar("kernel/logdet", slogdet(kernel)[-1], i)

    # norms
    writer.add_scalar("kernel:norm/fro", matrix_norm(kernel, ord="fro"), i)
    writer.add_scalar("kernel:norm/nuc", matrix_norm(kernel, ord="nuc"), i)
    writer.add_scalar("kernel:norm/-∞", matrix_norm(kernel, ord=-float("inf")), i)
    writer.add_scalar("kernel:norm/-2", matrix_norm(kernel, ord=-2), i)
    writer.add_scalar("kernel:norm/-1", matrix_norm(kernel, ord=-1), i)
    writer.add_scalar("kernel:norm/+1", matrix_norm(kernel, ord=+1), i)
    writer.add_scalar("kernel:norm/+2", matrix_norm(kernel, ord=+2), i)
    writer.add_scalar("kernel:norm/+∞", matrix_norm(kernel, ord=+float("inf")), i)

    # 2d -data is order of magnitude more expensive.
    if histograms:
        writer.add_image("kernel/values", torch.tanh(kernel), i, dataformats="HW")
        writer.add_figure("kernel/spectrum", plot_spectrum(kernel), i)
        writer.add_histogram("kernel/histogram", kernel, i)


def log_optimizer_state(
    i: int, writer: SummaryWriter, optimizer: Optimizer, histograms: bool = False
):
    """Log optimizer data.

    Parameters
    ----------
    i: int
    writer: SummaryWriter
    optimizer: Optimizer
    histograms: bool=False
    """
    # TODO: make this compatible with optimziers other than ADAM.
    LOGGER.warning("Optimizer logging only works with ADAM currently")
    variables = list(optimizer.state.keys())
    gradients = [w.grad for w in variables]
    moments_1 = [d["exp_avg"] for d in optimizer.state.values()]
    moments_2 = [d["exp_avg_sq"] for d in optimizer.state.values()]

    writer.add_scalar("optim:norms/variables", multi_norm(variables), i)
    writer.add_scalar("optim:norms/gradients", multi_norm(gradients), i)
    writer.add_scalar("optim:norms/moments_1", multi_norm(moments_1), i)
    writer.add_scalar("optim:norms/moments_2", multi_norm(moments_2), i)

    # 2d -data is order of magnitude more expensive.
    if histograms:
        for i, (w, g, a, b) in enumerate(
            zip(variables, gradients, moments_1, moments_2)
        ):
            writer.add_histogram(f"optim:variables/{i}", w, i)
            writer.add_histogram(f"optim:gradients/{i}", g, i)
            writer.add_histogram(f"optim:moments_1/{i}", a, i)
            writer.add_histogram(f"optim:moments_2/{i}", b, i)
