r"""Module Summary Line.

Module description
"""

from __future__ import annotations

__all__ = [
    "log_kernel_information",
    "log_optimizer_state",
    "log_model_state",
    "log_metrics",
]

import logging
from typing import Optional, Union

import torch
from torch import Tensor, jit, nn
from torch.linalg import cond, det, matrix_norm, matrix_rank, slogdet
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.losses import Loss
from tsdm.losses.functional import FunctionalLoss
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.plot import kernel_heatmap, plot_spectrum
from tsdm.util._norms import multi_norm

LOGGER = logging.getLogger(__name__)


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


@torch.no_grad()
def log_kernel_information(
    i: int,
    writer: SummaryWriter,
    kernel: Tensor,
    histograms: bool = False,
    prefix: Optional[str] = None,
):
    r"""Log kernel information.

    Parameters
    ----------
    i: int
    writer: SummaryWriter
    kernel: Tensor
    histograms: bool = False
    prefix: Optional[str] = None,
        Adds a prefix to the tensorboard logging path
    """
    assert len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1]
    prefix = f"{prefix}:kernel" if prefix is not None else "kernel"
    inf = float("inf")

    writer.add_scalar(f"{prefix}/skewpart", skewpart(kernel), i)
    writer.add_scalar(f"{prefix}/symmpart", symmpart(kernel), i)

    # general properties
    writer.add_scalar(f"{prefix}/det", det(kernel), i)
    writer.add_scalar(f"{prefix}/logdet", slogdet(kernel)[-1], i)
    writer.add_scalar(f"{prefix}/rank", matrix_rank(kernel), i)
    writer.add_scalar(f"{prefix}/trace", torch.trace(kernel), i)
    writer.add_scalar(f"{prefix}/cond", cond(kernel), i)

    # norms
    writer.add_scalar(f"{prefix}:norms/fro", matrix_norm(kernel, ord="fro"), i)
    writer.add_scalar(f"{prefix}:norms/nuc", matrix_norm(kernel, ord="nuc"), i)
    writer.add_scalar(f"{prefix}:norms/l-∞", matrix_norm(kernel, ord=-inf), i)
    writer.add_scalar(f"{prefix}:norms/l-2", matrix_norm(kernel, ord=-2), i)
    writer.add_scalar(f"{prefix}:norms/l-1", matrix_norm(kernel, ord=-1), i)
    writer.add_scalar(f"{prefix}:norms/l+1", matrix_norm(kernel, ord=+1), i)
    writer.add_scalar(f"{prefix}:norms/l+2", matrix_norm(kernel, ord=+2), i)
    writer.add_scalar(f"{prefix}:norms/l+∞", matrix_norm(kernel, ord=+inf), i)

    # 2d -data is order of magnitude more expensive.
    if histograms:
        writer.add_histogram(f"{prefix}/histogram", kernel, i)
        writer.add_image(f"{prefix}/values", kernel_heatmap(kernel, "CHW"), i)
        writer.add_figure(f"{prefix}/spectrum", plot_spectrum(kernel), i)


@torch.no_grad()
def log_optimizer_state(
    i: int,
    writer: SummaryWriter,
    optimizer: Optimizer,
    histograms: bool = False,
    prefix: Optional[str] = None,
):
    r"""Log optimizer data.

    Parameters
    ----------
    i: int
    writer: SummaryWriter
    optimizer: Optimizer
    histograms: bool=False
    prefix: Optional[str] = None,
    """
    prefix = f"{prefix}:optim" if prefix is not None else "optim"

    # TODO: make this compatible with optimziers other than ADAM.
    if not isinstance(optimizer, torch.optim.Adam):
        raise RuntimeError("Optimizer logging only works with ADAM currently")

    variables = list(optimizer.state.keys())
    gradients = [w.grad for w in variables]
    moments_1 = [d["exp_avg"] for d in optimizer.state.values()]
    moments_2 = [d["exp_avg_sq"] for d in optimizer.state.values()]

    writer.add_scalar(f"{prefix}:norms/variables", multi_norm(variables), i)
    writer.add_scalar(f"{prefix}:norms/gradients", multi_norm(gradients), i)
    writer.add_scalar(f"{prefix}:norms/moments_1", multi_norm(moments_1), i)
    writer.add_scalar(f"{prefix}:norms/moments_2", multi_norm(moments_2), i)

    # 2d -data is order of magnitude more expensive.
    if histograms:
        for j, (w, g, a, b) in enumerate(
            zip(variables, gradients, moments_1, moments_2)
        ):
            writer.add_histogram(f"{prefix}:variables/{j}", w, i)
            writer.add_histogram(f"{prefix}:gradients/{j}", g, i)
            writer.add_histogram(f"{prefix}:moments_1/{j}", a, i)
            writer.add_histogram(f"{prefix}:moments_2/{j}", b, i)


@torch.no_grad()
def log_model_state(
    i: int,
    writer: SummaryWriter,
    model: Model,
    histograms: bool = False,
    prefix: Optional[str] = None,
):
    """Log optimizer data.

    Parameters
    ----------
    i: int
    writer: SummaryWriter
    optimizer: Optimizer
    histograms: bool=False
    prefix: Optional[str] = None,
    """
    # TODO: make this compatible with optimziers other than ADAM.
    prefix = f"{prefix}:model" if prefix is not None else "model"
    variables = list(model.parameters())
    gradients = [w.grad for w in variables]

    writer.add_scalar(f"{prefix}:norms/variables", multi_norm(variables), i)
    writer.add_scalar(f"{prefix}:norms/gradients", multi_norm(gradients), i)

    # 2d -data is order of magnitude more expensive.
    if histograms:
        for i, (w, g) in enumerate(zip(variables, gradients)):
            writer.add_histogram(f"{prefix}:variables/{i}", w, i)
            writer.add_histogram(f"{prefix}:gradients/{i}", g, i)


@torch.no_grad()
def log_metrics(
    i: int,
    writer: SummaryWriter,
    targets: Tensor,
    predics: Tensor,
    metrics: Union[dict[str, Union[Loss, FunctionalLoss]], list[Loss]],
    prefix: Optional[str] = None,
):
    r"""Log each metric provided.

    Parameters
    ----------
    i: int,
    writer: SummaryWriter,
    targets: Tensor,
    predics: Tensor,
    metrics: dict[str, Union[Loss, FunctionalLoss]] or list[Loss]
    prefix: Optional[str] = None,
    """
    prefix = f"{prefix}:metrics" if prefix is not None else "metrics"

    if isinstance(metrics, list):
        for metric in metrics:
            assert issubclass(metric, nn.Module), "Metric must be nn.Module!"
            writer.add_scalar(
                f"{prefix}/{metric.__name__}", metric()(targets, predics), i
            )

    for name, metric in metrics.items():  # type: ignore
        if issubclass(metric, nn.Module):
            writer.add_scalar(f"{prefix}/{name}", metric()(targets, predics), i)
        else:
            writer.add_scalar(f"{prefix}/{name}", metric(targets, predics), i)
