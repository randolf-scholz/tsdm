r"""Logging Utility Functions."""

__all__ = [
    # Functions
    "compute_metrics",
    "log_kernel_information",
    "log_optimizer_state",
    "log_model_state",
    "log_metrics",
    "log_values",
    # Classes
    "StandardLogger",
]

import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypedDict, Union

import pandas as pd
import torch
from pandas import DataFrame, Index, MultiIndex
from torch import Tensor, nn
from torch.linalg import cond, matrix_norm, slogdet
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.notebook import tqdm

from tsdm.losses import Loss
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.plot import kernel_heatmap, plot_spectrum
from tsdm.util import (
    erank,
    mat_corr,
    multi_norm,
    reldist_diag,
    reldist_orth,
    reldist_skew,
    reldist_symm,
)


@torch.no_grad()
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
    histograms: Union[bool, int] = False,
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
    identifier = f"{prefix+':'*bool(prefix)}kernel{':'*bool(postfix)+postfix}"
    K = kernel
    assert len(K.shape) == 2 and K.shape[0] == K.shape[1]

    # relative distance to some matrix groups
    writer.add_scalar(f"{identifier}:reldist/diagonal", reldist_diag(K), i)
    writer.add_scalar(f"{identifier}:reldist/skew-symmetric", reldist_skew(K), i)
    writer.add_scalar(f"{identifier}:reldist/symmetric", reldist_symm(K), i)
    writer.add_scalar(f"{identifier}:reldist/orthogonal", reldist_orth(K), i)

    # general properties
    writer.add_scalar(f"{identifier}:linalg/logdet", slogdet(K)[-1], i)
    writer.add_scalar(f"{identifier}:linalg/erank", erank(K), i)
    writer.add_scalar(f"{identifier}:linalg/trace", torch.trace(K), i)
    writer.add_scalar(f"{identifier}:linalg/cond", cond(K), i)
    writer.add_scalar(f"{identifier}:linalg/mean-correlation", mat_corr(K), i)

    # norms
    writer.add_scalar(f"{identifier}:norms/fro", matrix_norm(K, ord="fro"), i)
    writer.add_scalar(f"{identifier}:norms/nuc", matrix_norm(K, ord="nuc"), i)
    writer.add_scalar(f"{identifier}:norms/l-∞", matrix_norm(K, ord=-float("inf")), i)
    writer.add_scalar(f"{identifier}:norms/l-2", matrix_norm(K, ord=-2), i)
    writer.add_scalar(f"{identifier}:norms/l-1", matrix_norm(K, ord=-1), i)
    writer.add_scalar(f"{identifier}:norms/l+1", matrix_norm(K, ord=+1), i)
    writer.add_scalar(f"{identifier}:norms/l+2", matrix_norm(K, ord=+2), i)
    writer.add_scalar(f"{identifier}:norms/l+∞", matrix_norm(K, ord=+float("inf")), i)

    # 2d-data is order of magnitude more expensive.
    if histograms and i % histograms == 0:
        writer.add_histogram(f"{identifier}:histogram", K, i)
        writer.add_image(f"{identifier}:heatmap", kernel_heatmap(K, "CHW"), i)
        writer.add_figure(f"{identifier}:spectrum", plot_spectrum(K), i)


def log_optimizer_state(
    i: int,
    /,
    writer: SummaryWriter,
    optimizer: Optimizer,
    *,
    histograms: Union[bool, int] = False,
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

    variables = list(optimizer.state.keys())
    gradients = [w.grad for w in variables]

    writer.add_scalar(f"{identifier}:norms/variables", multi_norm(variables), i)
    writer.add_scalar(f"{identifier}:norms/gradients", multi_norm(gradients), i)

    try:
        moments_1 = [d["exp_avg"] for d in optimizer.state.values()]
        moments_2 = [d["exp_avg_sq"] for d in optimizer.state.values()]
    except KeyError:
        moments_1 = []
        moments_2 = []
    else:
        writer.add_scalar(f"{identifier}:norms/moments_1", multi_norm(moments_1), i)
        writer.add_scalar(f"{identifier}:norms/moments_2", multi_norm(moments_2), i)

    if histograms and i % histograms == 0:
        for j, (w, g) in enumerate(zip(variables, gradients)):
            writer.add_histogram(f"{identifier}:variables/{j}", w, i)
            writer.add_histogram(f"{identifier}:gradients/{j}", g, i)

        for j, (a, b) in enumerate(zip(moments_1, moments_2)):
            writer.add_histogram(f"{identifier}:moments_1/{j}", a, i)
            writer.add_histogram(f"{identifier}:moments_2/{j}", b, i)


@torch.no_grad()
def log_model_state(
    i: int,
    /,
    writer: SummaryWriter,
    model: Model,
    *,
    histograms: Union[bool, int] = False,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log model data."""
    identifier = f"{prefix+':'*bool(prefix)}model{':'*bool(postfix)+postfix}"

    weights: dict[str, Tensor] = dict(model.named_parameters())
    grads: dict[str, Tensor] = {
        k: w.grad for k, w in weights.items() if w.grad is not None
    }

    if weights:
        weight_list = list(weights.values())
        writer.add_scalar(f"{identifier}:weights", multi_norm(weight_list), i)
    if grads:
        grad_list = list(grads.values())
        writer.add_scalar(f"{identifier}:gradients", multi_norm(grad_list), i)

    if histograms and i % histograms == 0:
        for key, weight in weights.items():
            writer.add_histogram(f"{identifier}:variables/{key}", weight, i)

        for key, gradient in grads.items():
            writer.add_histogram(f"{identifier}:gradients/{key}", gradient, i)


def log_metrics(
    i: int,
    /,
    writer: SummaryWriter,
    metrics: Union[dict[str, Loss], Sequence[str]],
    *,
    targets: Tensor,
    predics: Tensor,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    assert len(targets) == len(predics)
    assert isinstance(metrics, dict)
    values = compute_metrics(metrics, targets=targets, predics=predics)
    log_values(i, writer, values, prefix=prefix, postfix=postfix)


def log_values(
    i: int,
    /,
    writer: SummaryWriter,
    values: dict[str, Tensor],
    *,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    identifier = f"{prefix+':'*bool(prefix)}metrics{':'*bool(postfix)+postfix}"
    for key in values:
        writer.add_scalar(f"{identifier}/{key}", values[key], i)


class ResultTuple(NamedTuple):
    r"""Tuple of targets and predictions."""

    targets: Tensor
    predics: Tensor


class ResultDict(TypedDict):
    r"""Tuple of targets and predictions."""

    targets: Tensor
    predics: Tensor


@dataclass
class StandardLogger:
    r"""Logger for training and validation."""

    writer: SummaryWriter
    model: nn.Module
    metrics: dict[str, Callable[[Tensor, Tensor], Tensor]]
    optimizer: torch.optim.Optimizer
    dataloaders: Mapping[str, DataLoader]
    hparam_dict: dict[str, Any]
    predict_fn: Union[
        Callable[[nn.Module, tuple], ResultTuple],
        Callable[[nn.Module, tuple], ResultDict],
    ]
    checkpoint_dir: Path

    history: DataFrame = field(init=False)
    log_dir: Path = field(init=False)
    _warned_tuple: bool = False

    def __post_init__(self) -> None:
        """Initialize logger."""
        self.log_dir = self.writer.log_dir
        columns = MultiIndex.from_product([self.dataloaders, self.metrics])
        index = Index([], name="epoch", dtype=int)
        self.history = DataFrame(index=index, columns=columns, dtype="Float32")

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}"

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> ResultTuple:
        r"""Get all predictions for a dataloader."""
        predics = []
        targets = []
        for batch in tqdm(dataloader, "computing predictions", leave=False):
            result = self.predict_fn(self.model, batch)

            if isinstance(result, dict):
                targets.append(result["targets"])
                predics.append(result["predics"])
            elif isinstance(result, ResultTuple):
                targets.append(result.targets)
                predics.append(result.predics)
            else:
                if not self._warned_tuple:
                    self._warned_tuple = True
                    warnings.warn(
                        "We strongly recommend using NamedTuple or a dictionary"
                        " as the return type of the predict_fn. "
                        "Make sure that 'targets' is the first output and 'predics' is the second.",
                        UserWarning,
                    )
                targets.append(result[0])
                predics.append(result[1])

        predics = torch.cat(predics)
        targets = torch.cat(targets)
        return ResultTuple(targets=targets, predics=predics)

    def make_checkpoint(self, i: int, /) -> None:
        r"""Make checkpoint."""
        # create checkpoint
        torch.jit.save(
            self.model,
            self.checkpoint_dir / f"{self.model.__class__.__name__}-{i}",
        )
        torch.save(
            self.optimizer,
            self.checkpoint_dir / f"{self.optimizer.__class__.__name__}-{i}",
        )

    def log_metrics(
        self,
        i: int,
        /,
        *,
        targets: Tensor,
        predics: Tensor,
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        r"""Log metrics."""
        log_metrics(
            i,
            writer=self.writer,
            metrics=self.metrics,
            targets=targets,
            predics=predics,
            prefix=prefix,
            postfix=postfix,
        )

    def log_batch_end(self, i: int, *, targets: Tensor, predics: Tensor) -> None:
        r"""Log batch end."""
        self.log_metrics(i, targets=targets, predics=predics, postfix="batch")
        self.log_optimizer_state(i)

    def log_epoch_end(
        self,
        i: int,
        *,
        histograms: Union[bool, int] = True,
        kernel_information: Union[bool, int] = 1,
        model_state: Union[bool, int] = 10,
        optimizer_state: Union[bool, int] = False,
        make_checkpoint: Union[bool, int] = 10,
    ) -> None:
        r"""Log epoch end."""
        self.log_all_metrics(i)

        if kernel_information and i % kernel_information == 0:
            self.log_kernel_information(i, histograms=histograms)

        if model_state and i % model_state == 0:
            self.log_model_state(i, histograms=histograms)

        if optimizer_state and i % optimizer_state == 0:
            self.log_optimizer_state(i, histograms=histograms)

        if make_checkpoint and i % make_checkpoint == 0:
            self.make_checkpoint(i)

    def log_all_metrics(self, epoch: int, /) -> None:
        r"""Log all metrics for all dataloaders."""
        if epoch not in self.history.index:
            empty_row = DataFrame(
                index=[epoch], columns=self.history.columns, dtype="Float32"
            )
            self.history = pd.concat([self.history, empty_row], sort=True)

        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            values = compute_metrics(
                self.metrics, targets=result.targets, predics=result.predics
            )

            for metric, value in values.items():
                self.history.loc[epoch, (key, metric)] = value.cpu().item()

            log_values(
                epoch,
                writer=self.writer,
                values=values,
                postfix=key,
            )

    def log_hparams(self) -> None:
        r"""Log hyperparameters."""
        # find best epoch on the smoothed validation curve
        best_epochs = self.history.rolling(5, center=True).mean().idxmin()
        test_scores = {
            f"metrics:hparam/{key}": self.history.loc[idx, ("test", key)]
            for key, idx in best_epochs["valid"].items()
        }
        self.writer.add_hparams(
            hparam_dict=self.hparam_dict, metric_dict=test_scores, run_name="hparam"
        )
        print(f"{test_scores=} achieved by {self.hparam_dict=}")

        # Cleanup -remove when pytorch/issues/32651 is fixed
        for file in (self.log_dir / "hparam").iterdir():
            shutil.move(file, self.log_dir)
        (self.log_dir / "hparam").rmdir()

    def log_history(self, path: Optional[Path] = None) -> None:
        r"""Store history dataframe to file (default format: parquet)."""
        path = path if path is not None else self.log_dir / "history.parquet"
        self.history.to_parquet(path)

    def log_optimizer_state(
        self,
        i: int,
        /,
        *,
        histograms: Union[bool, int] = False,
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        r"""Log optimizer state."""
        log_optimizer_state(
            i,
            writer=self.writer,
            optimizer=self.optimizer,
            histograms=histograms,
            prefix=prefix,
            postfix=postfix,
        )

    def log_model_state(
        self,
        i: int,
        /,
        *,
        histograms: Union[bool, int] = False,
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        r"""Log model state."""
        log_model_state(
            i,
            writer=self.writer,
            model=self.model,
            histograms=histograms,
            prefix=prefix,
            postfix=postfix,
        )

    def log_kernel_information(
        self,
        i: int,
        /,
        *,
        histograms: Union[bool, int] = False,
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        """Log kernel information."""
        assert hasattr(self.model, "kernel") and isinstance(self.model.kernel, Tensor)
        log_kernel_information(
            i,
            writer=self.writer,
            kernel=self.model.kernel,
            histograms=histograms,
            prefix=prefix,
            postfix=postfix,
        )


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

#
# @dataclass
# class DefaultLogger:
#     r"""Default logger."""
#
#     writer: SummaryWriter
#     r"""The SummaryWriter Instance."""
#     model: Model
#     r"""The model instance."""
#     task: Task
#     r"""The task instance."""
#     optimizer: Optimizer
#     r"""The optimizer instance."""
#     encoder: Encoder
#     r"""The used pre_encoder."""
#     metrics: dict[str, Loss]
#     r"""The metrics that should be logged."""
#     num_epoch: int = 0
#     r"""The current batch number."""
#     num_batch: int = 0
#     r"""The current epoch number."""
#
#     # Lazy attributes
#     dataloaders: Mapping[str, DataLoader] = field(init=False)
#     r"""Pointer to the dataloaders associated with the task."""
#     history: dict[str, DataFrame] = field(init=False)
#     r"""Auto-updating DataFrame (similar to Keras)."""
#     LOGDIR: Path = field(init=False)
#     r"""The path where logs are stored."""
#     CHECKPOINTDIR: Path = field(init=False)
#     r"""The path where model checkpoints are stored."""
#
#     def __post_init__(self) -> None:
#         r"""Initialize the remaining attributes."""
#         self.KEYS = set(self.dataloaders)
#         self.dataloaders = self.task.dataloaders
#         self.LOGDIR = Path(self.writer.log_dir)
#         self.CHECKPOINTDIR = Path(self.writer.log_dir)
#
#         if self.history is None:
#             self.history = {
#                 key: DataFrame(columns=self.metrics) for key in self.KEYS | {"batch"}
#             }
#
#     @torch.no_grad()
#     def log_at_batch_end(self, *, targets: Tensor, predics: Tensor) -> None:
#         r"""Log metrics and optimizer state at the end of batch."""
#         self.num_batch += 1
#         values = compute_metrics(targets=targets, predics=predics, metrics=self.metrics)
#         log_metrics(
#             self.num_batch,
#             writer=self.writer,
#             metrics=self.metrics,
#             values=values,
#             prefix="batch",
#         )
#         log_optimizer_state(
#             self.num_batch, writer=self.writer, optimizer=self.optimizer, prefix="batch"
#         )
#         self.history["batch"].append(self._to_cpu(values))
#
#     @torch.no_grad()
#     def log_at_epoch_end(self, *, targets: Tensor, predics: Tensor) -> None:
#         r"""Log metric and optimizer state at the end of epoch."""
#         self.num_epoch += 1
#
#         log_optimizer_state(
#             self.num_epoch,
#             writer=self.writer,
#             optimizer=self.optimizer,
#             histograms=True,
#         )
#         # log_kernel_information(self.epoch, writer, model.system.kernel, histograms=True)
#
#         for key in self.dataloaders:
#             hist = compute_metrics(
#                 targets=targets, predics=predics, metrics=self.metrics
#             )
#             log_metrics(
#                 self.num_epoch, self.writer, self.metrics, values=hist, prefix=key
#             )
#             self.history[key].append(self._to_cpu(hist))
#
#     @staticmethod
#     def _to_cpu(scalar_dict: dict[str, Tensor]) -> dict[str, float]:
#         return {key: scalar.item() for key, scalar in scalar_dict.items()}
#
#     def save_checkpoint(self) -> None:
#         r"""Save the hyperparameter combination with validation loss."""
#         torch.save(
#             {
#                 "optimizer": self.optimizer,
#                 "epoch": self.num_epoch,
#                 "batch": self.num_batch,
#             },
#             self.CHECKPOINTDIR.joinpath(
#                 f"{self.optimizer.__class__.__name__}-{self.num_epoch}"
#             ),
#         )

# def log_hyperparameters(self):
#     r"""Save the hyperparameter combination with validation loss."""
#     ...
