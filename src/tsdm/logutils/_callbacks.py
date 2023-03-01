r"""Logging Utility Functions.

We allow the user to define callbacks for logging, which are called at the end of
batches/epochs.

A callback is a function that takes the following arguments:

- ``i``: The current iteration/epoch.
- ``writer``: The ``SummaryWriter`` instance.
- ``logged_object``: The object that is being logged.
    - what if we need to pass multiple objects?
- when used in callbacks, the
- Should a callback be self-contained?
- Does the callback need to reference the Logger it is being called from?

Idea: We want to so something like this:

.. code-block:: python

    logger = Logger(...)
    logger.add_callback(log_loss, on="batch")

How do we ensure logging is fast?

- loggers should not recompute things that have already been computed
    - loggers need access to existing results

Examples
--------
We want to log the left-inverse residual of the linodenet.
Sometimes, this is also used as a regularization term.
therefore the logging function either has to:

- compute R and log it
- use the existing R and log it

Typical things the loggers need access to include:

- the current iteration
- the current loss
- the current model
- the current optimizer
- the current data
    - the current predictions
    - the current targets
- the current metrics
"""

__all__ = [
    # Functions
    "compute_metrics",
    "log_kernel_information",
    "log_optimizer_state",
    "log_model_state",
    "log_metrics",
    "log_values",
    # Classes
    # Protocols
]

import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import yaml
from matplotlib.pyplot import Figure
from matplotlib.pyplot import close as close_figure
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.linalg import (
    col_corr,
    erank,
    logarithmic_norm,
    matrix_norm,
    multi_norm,
    operator_norm,
    reldist_diag,
    reldist_orth,
    reldist_skew,
    reldist_symm,
    row_corr,
    schatten_norm,
)
from tsdm.metrics import Loss
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.types.variables import Any_contra as T_contra
from tsdm.viz import center_axes, kernel_heatmap, plot_spectrum, rasterize


@runtime_checkable
class Callback(Protocol):
    """Protocol for callbacks."""

    def __call__(self, i: int, /) -> None:
        """Log something at time index i."""


class BaseCallback(ABC):
    """Base class for callbacks."""

    writer: SummaryWriter
    """The ``SummaryWriter`` instance."""
    logged_object: Any
    """The object that is being logged."""
    identifier: str
    """The identifier of the logged object."""

    def __init__(
        self,
        writer: SummaryWriter,
        /,
        key: str = "",
        name: str = "metrics",
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        self.writer = writer
        self.identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"

    @abstractmethod
    def __call__(self, i: int, /) -> None:
        """Log something at the end of a batch/epoch."""


@torch.no_grad()
def compute_metrics(
    metrics: list | Mapping[str, Any], /, *, targets: Tensor, predics: Tensor
) -> dict[str, Tensor]:
    r"""Compute multiple metrics."""
    results: dict[str, Tensor] = {}
    if isinstance(metrics, list):
        for metric in metrics:
            if issubclass(metric, nn.Module):
                results[metric.__name__] = metric()(targets, predics)
            elif callable(metric) | isinstance(metric, nn.Module):
                results[metric.__name__] = metric(targets, predics)
            else:
                raise ValueError(metric)
        return results

    for name, metric in metrics.items():
        if callable(metric) | isinstance(metric, nn.Module):
            results[name] = metric(targets, predics)
        elif isinstance(metric, type) and issubclass(metric, nn.Module):
            results[name] = metric()(targets, predics)
        else:
            raise ValueError(f"{type(metric)=} not understood!")
    return results


@torch.no_grad()
def log_kernel_information(
    i: int,
    /,
    writer: SummaryWriter,
    kernel: Tensor,
    *,
    log_figures: bool = True,
    log_scalars: bool = True,
    # individual switches
    log_distances: bool = True,
    log_heatmap: bool = True,
    log_linalg: bool = True,
    log_norms: bool = True,
    log_scaled_norms: bool = True,
    log_spectrum: bool = True,
    name: str = "kernel",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log kernel information.

    Set option to true to log every epoch.
    Set option to an integer to log whenever ``i % log_interval == 0``.
    """
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"
    K = kernel
    assert len(K.shape) == 2 and K.shape[0] == K.shape[1]

    log = writer.add_scalar
    ω = float("inf")

    if log_scalars and log_distances:
        # fmt: off
        # relative distance to some matrix groups
        log(f"{identifier}:distances/diagonal_(relative)",       reldist_diag(K), i)
        log(f"{identifier}:distances/skew-symmetric_(relative)", reldist_skew(K), i)
        log(f"{identifier}:distances/symmetric_(relative)",      reldist_symm(K), i)
        log(f"{identifier}:distances/orthogonal_(relative)",     reldist_orth(K), i)
        # fmt: on

    if log_scalars and log_linalg:
        # fmt: off
        # general properties
        log(f"{identifier}:linalg/condition-number", torch.linalg.cond(K), i)
        log(f"{identifier}:linalg/correlation-cols", col_corr(K), i)
        log(f"{identifier}:linalg/correlation-rows", row_corr(K), i)
        log(f"{identifier}:linalg/effective-rank",   erank(K), i)
        log(f"{identifier}:linalg/logdet",           torch.linalg.slogdet(K)[-1], i)
        log(f"{identifier}:linalg/mean",             torch.mean(K), i)
        log(f"{identifier}:linalg/stdev",            torch.std(K), i)
        log(f"{identifier}:linalg/trace",            torch.trace(K), i)
        # fmt: on

    if log_scalars and log_norms:
        # fmt: off
        # Matrix norms
        log(f"{identifier}:norms/matrix_+∞_(max_absolut_value)", matrix_norm(K, p=+ω), i)
        log(f"{identifier}:norms/matrix_+2_(sum_squared_value)", matrix_norm(K, p=+2), i)
        log(f"{identifier}:norms/matrix_+1_(sum_absolut_value)", matrix_norm(K, p=+1), i)
        log(f"{identifier}:norms/matrix_-1_(sum_inverse_value)", matrix_norm(K, p=-1), i)
        log(f"{identifier}:norms/matrix_-2_(sum_isquare_value)", matrix_norm(K, p=-2), i)
        log(f"{identifier}:norms/matrix_-∞_(min_absolut_value)", matrix_norm(K, p=-ω), i)
        # Logarithmic norms
        log(f"{identifier}:norms/logarithmic_+∞_(max_row_sum)", logarithmic_norm(K, p=+ω), i)
        log(f"{identifier}:norms/logarithmic_+2_(max_eig_val)", logarithmic_norm(K, p=+2), i)
        log(f"{identifier}:norms/logarithmic_+1_(max_col_sum)", logarithmic_norm(K, p=+1), i)
        log(f"{identifier}:norms/logarithmic_-1_(min_col_sum)", logarithmic_norm(K, p=-1), i)
        log(f"{identifier}:norms/logarithmic_-2_(min_eig_val)", logarithmic_norm(K, p=-2), i)
        log(f"{identifier}:norms/logarithmic_-∞_(min_row_sum)", logarithmic_norm(K, p=-ω), i)
        # Operator norms
        log(f"{identifier}:norms/operator_+∞_(max_row_sum)", operator_norm(K, p=+ω), i)
        log(f"{identifier}:norms/operator_+2_(max_σ_value)", operator_norm(K, p=+2), i)
        log(f"{identifier}:norms/operator_+1_(max_col_sum)", operator_norm(K, p=+1), i)
        log(f"{identifier}:norms/operator_-1_(min_col_sum)", operator_norm(K, p=-1), i)
        log(f"{identifier}:norms/operator_-2_(min_σ_value)", operator_norm(K, p=-2), i)
        log(f"{identifier}:norms/operator_-∞_(min_row_sum)", operator_norm(K, p=-ω), i)
        # Schatten norms
        log(f"{identifier}:norms/schatten_+∞_(max_absolut_σ)", schatten_norm(K, p=+ω), i)
        log(f"{identifier}:norms/schatten_+2_(sum_squared_σ)", schatten_norm(K, p=+2), i)
        log(f"{identifier}:norms/schatten_+1_(sum_absolut_σ)", schatten_norm(K, p=+1), i)
        log(f"{identifier}:norms/schatten_-1_(sum_inverse_σ)", schatten_norm(K, p=-1), i)
        log(f"{identifier}:norms/schatten_-2_(sum_isquare_σ)", schatten_norm(K, p=-2), i)
        log(f"{identifier}:norms/schatten_-∞_(min_absolut_σ)", schatten_norm(K, p=-ω), i)
        # fmt: on

    if log_scalars and log_scaled_norms:
        # fmt: off
        # Matrix norms
        log(f"{identifier}:norms-scaled/matrix_+∞_(max_absolut_value)", matrix_norm(K, p=+ω, scaled=True), i)
        log(f"{identifier}:norms-scaled/matrix_+2_(sum_squared_value)", matrix_norm(K, p=+2, scaled=True), i)
        log(f"{identifier}:norms-scaled/matrix_+1_(sum_absolut_value)", matrix_norm(K, p=+1, scaled=True), i)
        log(f"{identifier}:norms-scaled/matrix_-1_(sum_inverse_value)", matrix_norm(K, p=-1, scaled=True), i)
        log(f"{identifier}:norms-scaled/matrix_-2_(sum_isquare_value)", matrix_norm(K, p=-2, scaled=True), i)
        log(f"{identifier}:norms-scaled/matrix_-∞_(min_absolut_value)", matrix_norm(K, p=-ω, scaled=True), i)
        # Logarithmic norms
        log(f"{identifier}:norms-scaled/logarithmic_+∞_(max_row_sum)", logarithmic_norm(K, p=+ω, scaled=True), i)
        log(f"{identifier}:norms-scaled/logarithmic_+2_(max_eig_val)", logarithmic_norm(K, p=+2, scaled=True), i)
        log(f"{identifier}:norms-scaled/logarithmic_+1_(max_col_sum)", logarithmic_norm(K, p=+1, scaled=True), i)
        log(f"{identifier}:norms-scaled/logarithmic_-1_(min_col_sum)", logarithmic_norm(K, p=-1, scaled=True), i)
        log(f"{identifier}:norms-scaled/logarithmic_-2_(min_eig_val)", logarithmic_norm(K, p=-2, scaled=True), i)
        log(f"{identifier}:norms-scaled/logarithmic_-∞_(min_row_sum)", logarithmic_norm(K, p=-ω, scaled=True), i)
        # Operator norms
        log(f"{identifier}:norms-scaled/operator_+∞_(max_row_sum)", operator_norm(K, p=+ω, scaled=True), i)
        log(f"{identifier}:norms-scaled/operator_+2_(max_σ_value)", operator_norm(K, p=+2, scaled=True), i)
        log(f"{identifier}:norms-scaled/operator_+1_(max_col_sum)", operator_norm(K, p=+1, scaled=True), i)
        log(f"{identifier}:norms-scaled/operator_-1_(min_col_sum)", operator_norm(K, p=-1, scaled=True), i)
        log(f"{identifier}:norms-scaled/operator_-2_(min_σ_value)", operator_norm(K, p=-2, scaled=True), i)
        log(f"{identifier}:norms-scaled/operator_-∞_(min_row_sum)", operator_norm(K, p=-ω, scaled=True), i)
        # Schatten norms
        log(f"{identifier}:norms-scaled/schatten_+∞_(max_absolut_σ)", schatten_norm(K, p=+ω, scaled=True), i)
        log(f"{identifier}:norms-scaled/schatten_+2_(sum_squared_σ)", schatten_norm(K, p=+2, scaled=True), i)
        log(f"{identifier}:norms-scaled/schatten_+1_(sum_absolut_σ)", schatten_norm(K, p=+1, scaled=True), i)
        log(f"{identifier}:norms-scaled/schatten_-1_(sum_inverse_σ)", schatten_norm(K, p=-1, scaled=True), i)
        log(f"{identifier}:norms-scaled/schatten_-2_(sum_isquare_σ)", schatten_norm(K, p=-2, scaled=True), i)
        log(f"{identifier}:norms-scaled/schatten_-∞_(min_absolut_σ)", schatten_norm(K, p=-ω, scaled=True), i)
        # fmt: on

    # 2d-data is order of magnitude more expensive.
    if log_figures and log_heatmap:
        writer.add_image(f"{identifier}:heatmap", kernel_heatmap(K, "CHW"), i)

    if log_figures and log_spectrum:
        spectrum: Figure = plot_spectrum(K)
        spectrum = center_axes(spectrum)
        image = rasterize(spectrum, w=2, h=2, px=512, py=512)
        close_figure(spectrum)  # Important!
        writer.add_image(f"{identifier}:spectrum", image, i, dataformats="HWC")


def log_optimizer_state(
    i: int,
    /,
    writer: SummaryWriter,
    optimizer: Optimizer,
    *,
    log_histograms: bool = True,
    log_scalars: bool = True,
    loss: Optional[float | Tensor] = None,
    name: str = "optimizer",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log optimizer data under ``prefix:optimizer:postfix/``."""
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"

    variables = list(optimizer.state.keys())
    gradients = [w.grad for w in variables]
    moments_1 = [d["exp_avg"] for d in optimizer.state.values() if "exp_avg" in d]
    moments_2 = [d["exp_avg_sq"] for d in optimizer.state.values() if "exp_avg_sq" in d]
    log = writer.add_scalar

    if log_scalars:
        log(f"{identifier}/model-gradients", multi_norm(gradients), i)
        log(f"{identifier}/model-variables", multi_norm(variables), i)
        if loss is not None:
            log(f"{identifier}/loss", loss, i)
        if moments_1:
            log(f"{identifier}/param-moments_1", multi_norm(moments_1), i)
        if moments_2:
            log(f"{identifier}/param-moments_2", multi_norm(moments_2), i)

    if log_histograms:
        for j, (w, g) in enumerate(zip(variables, gradients)):
            if not w.numel():
                continue
            writer.add_histogram(f"{identifier}:model-variables/{j}", w, i)
            writer.add_histogram(f"{identifier}:model-gradients/{j}", g, i)

        for j, (a, b) in enumerate(zip(moments_1, moments_2)):
            if not a.numel():
                continue
            writer.add_histogram(f"{identifier}:param-moments_1/{j}", a, i)
            writer.add_histogram(f"{identifier}:param-moments_2/{j}", b, i)


@torch.no_grad()
def log_model_state(
    i: int,
    /,
    writer: SummaryWriter,
    model: Model,
    *,
    log_histograms: bool = True,
    log_scalars: bool = True,
    name: str = "model",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log model data."""
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"

    variables: dict[str, Tensor] = dict(model.named_parameters())
    gradients: dict[str, Tensor] = {
        k: w.grad for k, w in variables.items() if w.grad is not None
    }

    if log_scalars:
        writer.add_scalar(f"{identifier}/gradients", multi_norm(variables), i)
        writer.add_scalar(f"{identifier}/variables", multi_norm(gradients), i)

    if log_histograms:
        for key, weight in variables.items():
            if not weight.numel():
                continue
            writer.add_histogram(f"{identifier}:variables/{key}", weight, i)

        for key, gradient in gradients.items():
            if not gradient.numel():
                continue
            writer.add_histogram(f"{identifier}:gradients/{key}", gradient, i)


@dataclass
class KernelCallback(BaseCallback):
    """Callback to log kernel information to tensorboard."""

    writer: SummaryWriter
    kernel: Tensor

    _: KW_ONLY

    log_figures: bool = True
    log_scalars: bool = True

    # individual switches
    log_distances: bool = True
    log_heatmap: bool = True
    log_linalg: bool = True
    log_norms: bool = True
    log_scaled_norms: bool = True
    log_spectrum: bool = True

    name: str = "kernel"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, i: int) -> None:
        log_kernel_information(i, **asdict(self))


@dataclass
class OptimizerCallback(BaseCallback):
    writer: SummaryWriter
    optimizer: Optimizer
    _: KW_ONLY
    log_histograms: bool = True
    log_scalars: bool = True
    loss: Optional[float | Tensor] = None
    name: str = "optimizer"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, i: int, /) -> None:
        log_optimizer_state(i, **asdict(self))


@dataclass
class ModelCallback(BaseCallback):
    """Callback to log model information to tensorboard."""

    writer: SummaryWriter
    model: Model

    _: KW_ONLY

    log_histograms: bool = True
    log_scalars: bool = True

    name: str = "model"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, i: int) -> None:
        log_model_state(i, **asdict(self))


def log_scalar(
    i: int,
    /,
    writer: SummaryWriter,
    value: Tensor,
    *,
    key: str = "",
    name: str = "metrics",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log a single scalar value."""
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"
    writer.add_scalar(f"{identifier}/{key}", value, i)


def log_values(
    i: int,
    /,
    writer: SummaryWriter,
    values: Mapping[str, Tensor],
    *,
    key: str = "",
    name: str = "metrics",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"
    for metric in values:
        writer.add_scalar(f"{identifier}:{metric}/{key}", values[metric], i)


def log_metrics(
    i: int,
    /,
    writer: SummaryWriter,
    metrics: Sequence[str] | Mapping[str, Loss],
    *,
    targets: Tensor,
    predics: Tensor,
    key: str = "",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    assert len(targets) == len(predics)
    assert isinstance(metrics, dict)
    values = compute_metrics(metrics, targets=targets, predics=predics)
    log_values(i, writer, values, key=key, prefix=prefix, postfix=postfix)


def make_checkpoint(i: int, /, path: Path, **objects) -> None:
    """Save checkpoints of given paths."""
    path = path / f"{i}"
    path.mkdir(parents=True, exist_ok=True)

    for name, obj in objects.items():
        match obj:
            case torch.jit.ScriptModule():
                torch.jit.save(obj, path / name)
            case torch.nn.Module():
                torch.save(obj, path / name)
            case torch.optim.Optimizer():
                torch.save(obj, path / name)
            case torch.optim.lr_scheduler._LRScheduler():
                torch.save(obj, path / name)
            case dict() | list() | tuple() | set() | str() | int() | float() | None:
                with open(path / f"{name}.yaml", "w", encoding="utf8") as file:
                    yaml.safe_dump(obj, file)
            case _:
                with open(path / f"{name}.pickle", "wb") as file:
                    pickle.dump(obj, file)


@dataclass
class CheckpointCallback(BaseCallback):
    """Callback to save checkpoints."""

    path: Path
    objects: dict[str, object]

    def __call__(self, i: int) -> None:
        make_checkpoint(i, self.path, **self.objects)
