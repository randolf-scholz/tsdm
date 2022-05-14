r"""Logging Utility Functions."""

__all__ = [
    # Functions
    "compute_metrics",
    "log_kernel_information",
    "log_optimizer_state",
    "log_model_state",
    "log_metrics",
]

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, overload

import torch
from pandas import DataFrame
from torch import Tensor, nn
from torch.linalg import cond, det, matrix_norm, matrix_rank, slogdet
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.encoders import Encoder
from tsdm.losses import Loss
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.plot import kernel_heatmap, plot_spectrum
from tsdm.tasks import Task
from tsdm.util import multi_norm, relsize_skewpart, relsize_symmpart

__logger__ = logging.getLogger(__name__)


@torch.no_grad()
def log_kernel_information(
    i: int,
    *,
    writer: SummaryWriter,
    kernel: Tensor,
    histograms: bool = False,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log kernel information.

    Parameters
    ----------
    i: int
    writer: SummaryWriter
    kernel: Tensor
    histograms: bool, default False
    prefix: str, default ""
    postfix: str, default ""
    """
    assert len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1]

    identifier = f"{prefix+':'*bool(prefix)}kernel{':'*bool(postfix)+postfix}"

    inf = float("inf")

    writer.add_scalar(f"{identifier}/skewpart", relsize_skewpart(kernel), i)
    writer.add_scalar(f"{identifier}/symmpart", relsize_symmpart(kernel), i)

    # general properties
    writer.add_scalar(f"{identifier}/det", det(kernel), i)
    writer.add_scalar(f"{identifier}/logdet", slogdet(kernel)[-1], i)
    writer.add_scalar(f"{identifier}/rank", matrix_rank(kernel), i)
    writer.add_scalar(f"{identifier}/trace", torch.trace(kernel), i)
    writer.add_scalar(f"{identifier}/cond", cond(kernel), i)

    # norms
    writer.add_scalar(f"{identifier}:norms/fro", matrix_norm(kernel, ord="fro"), i)
    writer.add_scalar(f"{identifier}:norms/nuc", matrix_norm(kernel, ord="nuc"), i)
    writer.add_scalar(f"{identifier}:norms/l-∞", matrix_norm(kernel, ord=-inf), i)
    writer.add_scalar(f"{identifier}:norms/l-2", matrix_norm(kernel, ord=-2), i)
    writer.add_scalar(f"{identifier}:norms/l-1", matrix_norm(kernel, ord=-1), i)
    writer.add_scalar(f"{identifier}:norms/l+1", matrix_norm(kernel, ord=+1), i)
    writer.add_scalar(f"{identifier}:norms/l+2", matrix_norm(kernel, ord=+2), i)
    writer.add_scalar(f"{identifier}:norms/l+∞", matrix_norm(kernel, ord=+inf), i)

    # 2d-data is order of magnitude more expensive.
    if histograms:
        writer.add_histogram(f"{prefix}/histogram", kernel, i)
        writer.add_image(f"{prefix}/values", kernel_heatmap(kernel, "CHW"), i)
        writer.add_figure(f"{prefix}/spectrum", plot_spectrum(kernel), i)


@torch.no_grad()
def log_optimizer_state(
    i: int,
    *,
    writer: SummaryWriter,
    optimizer: Optimizer,
    histograms: bool = False,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log optimizer data.

    Parameters
    ----------
    i: int
    writer: SummaryWriter
    optimizer: Optimizer
    histograms: bool=False
    prefix: str = ""
    postfix: str = ""
    """
    identifier = f"{prefix+':'*bool(prefix)}optim{':'*bool(postfix)+postfix}"

    # TODO: make this compatible with optimizers other than ADAM.
    if not isinstance(optimizer, torch.optim.Adam):
        warnings.warn(
            "Optimizer logging only works with ADAM currently", RuntimeWarning
        )

    variables = list(optimizer.state.keys())
    gradients = [w.grad for w in variables]
    try:
        moments_1 = [d["exp_avg"] for d in optimizer.state.values()]
        moments_2 = [d["exp_avg_sq"] for d in optimizer.state.values()]
    except KeyError:
        pass

    writer.add_scalar(f"{identifier}:norms/variables", multi_norm(variables), i)
    writer.add_scalar(f"{identifier}:norms/gradients", multi_norm(gradients), i)
    try:
        writer.add_scalar(f"{identifier}:norms/moments_1", multi_norm(moments_1), i)
        writer.add_scalar(f"{identifier}:norms/moments_2", multi_norm(moments_2), i)
    except NameError:
        pass

    # NOTE: 2d-data is an order of magnitude more expensive.
    if histograms:
        for j, (w, g) in enumerate(zip(variables, gradients)):
            writer.add_histogram(f"{identifier}:variables/{j}", w, i)
            writer.add_histogram(f"{identifier}:gradients/{j}", g, i)
        try:
            for j, (a, b) in enumerate(zip(moments_1, moments_2)):
                writer.add_histogram(f"{identifier}:moments_1/{j}", a, i)
                writer.add_histogram(f"{identifier}:moments_2/{j}", b, i)
        except NameError:
            pass


@torch.no_grad()
def log_model_state(
    i: int,
    *,
    writer: SummaryWriter,
    model: Model,
    histograms: bool = False,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log optimizer data.

    Parameters
    ----------
    i: current step
    writer: SummaryWriter
    model: Model
    histograms: whether to create histograms
    prefix: e.g. prefix:model:postfix
    postfix: e.g. prefix:model:postfix
    """
    # TODO: make this compatible with optimizers other than ADAM.
    identifier = f"{prefix+':'*bool(prefix)}model{':'*bool(postfix)+postfix}"

    variables = list(model.parameters())
    gradients = [w.grad for w in variables]

    writer.add_scalar(f"{identifier}:norms/variables", multi_norm(variables), i)
    writer.add_scalar(f"{identifier}:norms/gradients", multi_norm(gradients), i)

    # 2d -data is order of magnitude more expensive.
    if histograms:
        for j, (w, g) in enumerate(zip(variables, gradients)):
            writer.add_histogram(f"{identifier}:variables/{j}", w, i)
            writer.add_histogram(f"{identifier}:gradients/{j}", g, i)


# user can provide targets and predics
@overload
def log_metrics(
    i: int,
    writer: SummaryWriter,
    metrics: dict[str, Loss],
    *,
    targets: Tensor,
    predics: Tensor,
    prefix: Optional[str] = None,
) -> None:
    ...


# user can provide values by dict
@overload
def log_metrics(
    i: int,
    writer: SummaryWriter,
    metrics: dict[str, Loss],
    *,
    values: dict[str, Tensor],
    prefix: Optional[str] = None,
) -> None:
    ...


# user can provide values by list
@overload
def log_metrics(
    i: int,
    writer: SummaryWriter,
    metrics: Sequence[str],
    *,
    values: Sequence[Tensor],
    prefix: Optional[str] = None,
) -> None:
    ...


@torch.no_grad()  # type: ignore[misc]
def log_metrics(
    i: int,
    writer: SummaryWriter,
    metrics: Union[dict[str, Loss], Sequence[str]],
    *,
    targets: Optional[Tensor] = None,
    predics: Optional[Tensor] = None,
    values: Optional[Union[dict[str, Tensor], Sequence[Tensor]]] = None,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once.

    Parameters
    ----------
    i: int,
    writer: SummaryWriter,
    metrics: dict[str, ModularLoss]
    *
    targets: Optional[Tensor], default None
    predics: Optional[Tensor], default None
    values: Optional[Tensor], default None
    prefix: str, default ""
    postfix: str, default ""
    """
    identifier = f"{prefix+':'*bool(prefix)}metrics{':'*bool(postfix)+postfix}"

    if values is None:
        assert (
            targets is not None and predics is not None
        ), "values and (targets, predics) are mutually exclusive"
        assert len(targets) == len(predics)
        assert isinstance(metrics, dict)
        values = compute_metrics(metrics, targets=targets, predics=predics)
    elif isinstance(values, Mapping) and isinstance(metrics, Mapping):
        assert (
            targets is None and predics is None
        ), "values and (targets, predics) are mutually exclusive"
        assert set(values.keys()) == set(metrics.keys())
    elif isinstance(values, Sequence) and isinstance(metrics, Sequence):
        assert (
            targets is None and predics is None
        ), "values and (targets, predics) are mutually exclusive"
        assert len(values) == len(metrics)
        values = dict(zip(metrics, values))
    else:
        raise ValueError(f"{values=} not understood!")

    for key in metrics:
        score = values[key]
        if not torch.isfinite(score).all():
            HAS_NAN = torch.isnan(score).any().item()
            HAS_INF = torch.isinf(score).any().item()
            raise RuntimeError(f"Model collapsed! {HAS_NAN=} {HAS_INF=}")
        writer.add_scalar(f"{identifier}/{key}", score, i)


def compute_metrics(
    metrics: Union[dict[str, Any], list[Any]], *, targets: Tensor, predics: Tensor
) -> dict[str, Tensor]:
    r"""Compute multiple metrics.

    Parameters
    ----------
    metrics: Union[dict[str, Any], list[Any]]
    targets: Tensor (keyword-only)
    predics: Tensor (keyword-only)

    Returns
    -------
    dict[str, Tensor]
        Name / Metric pairs
    """
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
        if isinstance(metric, nn.Module):
            results[name] = metric(targets, predics)
        elif isinstance(metric, type) and issubclass(metric, nn.Module):
            results[name] = metric()(targets, predics)
        else:
            raise ValueError(f"{type(metric)=} not understood!")
    return results


# @torch.no_grad()
# def get_all_predictions(model, dataloader):
#     r"""Get all predictions from a model."""
#     Y, Ŷ = [], []
#     for batch in tqdm(dataloader, leave=False):
#         # getting targets -> task / model
#         # getting predics -> task / model
#         OBS_HORIZON = 32
#         times, inputs, targets = prep_batch(batch, OBS_HORIZON)
#         outputs, _ = model(times, inputs)  # here we should apply the decoder.
#         predics = outputs[:, OBS_HORIZON:, -1]
#         Y.append(targets)
#         Ŷ.append(predics)
#
#     return torch.cat(Y, dim=0), torch.cat(Ŷ, dim=0)
#
#
# @jit.script  # This should be the pre_encoder
# def prep_batch(
#     batch: tuple[Tensor, Tensor, Tensor], observation_horizon: int
# ) -> tuple[Tensor, Tensor, Tensor]:
#     r"""Prepare a batch for training."""
#     T, X, Y = batch
#     timestamps = T
#     targets = Y[..., observation_horizon:].clone()
#     Y[..., observation_horizon:] = float("nan")  # mask future
#     X[..., observation_horizon:, :] = float("nan")  # mask future
#     observations = torch.cat([X, Y.unsqueeze(-1)], dim=-1)
#     inputs = (timestamps, observations)
#     return inputs, targets


@dataclass
class DefaultLogger:
    r"""Default logger."""

    writer: SummaryWriter
    r"""The SummaryWriter Instance."""
    model: Model
    r"""The model instance."""
    task: Task
    r"""The task instance."""
    optimizer: Optimizer
    r"""The optimizer instance."""
    encoder: Encoder
    r"""The used pre_encoder."""
    metrics: dict[str, Loss]
    r"""The metrics that should be logged."""
    num_epoch: int = 0
    r"""The current batch number."""
    num_batch: int = 0
    r"""The current epoch number."""

    # Lazy attributes
    dataloaders: Mapping[str, DataLoader] = field(init=False)
    r"""Pointer to the dataloaders associated with the task."""
    history: dict[str, DataFrame] = field(init=False)
    r"""Auto-updating DataFrame (similar to Keras)."""
    LOGDIR: Path = field(init=False)
    r"""The path where logs are stored."""
    CHECKPOINTDIR: Path = field(init=False)
    r"""The path where model checkpoints are stored."""

    def __post_init__(self) -> None:
        r"""Initialize the remaining attributes."""
        self.KEYS = set(self.dataloaders)
        self.dataloaders = self.task.dataloaders
        self.LOGDIR = Path(self.writer.log_dir)
        self.CHECKPOINTDIR = Path(self.writer.log_dir)

        if self.history is None:
            self.history = {
                key: DataFrame(columns=self.metrics) for key in self.KEYS | {"batch"}
            }

    @torch.no_grad()
    def log_at_batch_end(self, *, targets: Tensor, predics: Tensor) -> None:
        r"""Log metrics and optimizer state at the end of batch."""
        self.num_batch += 1
        values = compute_metrics(targets=targets, predics=predics, metrics=self.metrics)
        log_metrics(
            self.num_batch,
            writer=self.writer,
            metrics=self.metrics,
            values=values,
            prefix="batch",
        )
        log_optimizer_state(
            self.num_batch, writer=self.writer, optimizer=self.optimizer, prefix="batch"
        )
        self.history["batch"].append(self._to_cpu(values))

    @torch.no_grad()
    def log_at_epoch_end(self, *, targets: Tensor, predics: Tensor) -> None:
        r"""Log metric and optimizer state at the end of epoch."""
        self.num_epoch += 1

        log_optimizer_state(
            self.num_epoch,
            writer=self.writer,
            optimizer=self.optimizer,
            histograms=True,
        )
        # log_kernel_information(self.epoch, writer, model.system.kernel, histograms=True)

        for key in self.dataloaders:
            hist = compute_metrics(
                targets=targets, predics=predics, metrics=self.metrics
            )
            log_metrics(
                self.num_epoch, self.writer, self.metrics, values=hist, prefix=key
            )
            self.history[key].append(self._to_cpu(hist))

    @staticmethod
    def _to_cpu(scalar_dict: dict[str, Tensor]) -> dict[str, float]:
        return {key: scalar.item() for key, scalar in scalar_dict.items()}

    def save_checkpoint(self) -> None:
        r"""Save the hyperparameter combination with validation loss."""
        torch.save(
            {
                "optimizer": self.optimizer,
                "epoch": self.num_epoch,
                "batch": self.num_batch,
            },
            self.CHECKPOINTDIR.joinpath(
                f"{self.optimizer.__class__.__name__}-{self.num_epoch}"
            ),
        )

    # def log_hyperparameters(self):
    #     r"""Save the hyperparameter combination with validation loss."""
    #     ...