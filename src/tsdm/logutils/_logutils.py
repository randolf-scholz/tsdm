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
import yaml
from matplotlib.pyplot import Figure
from matplotlib.pyplot import close as close_figure
from pandas import DataFrame, Index, MultiIndex
from torch import Tensor, nn
from torch.linalg import cond, matrix_norm, slogdet
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.linalg import (
    col_corr,
    erank,
    multi_norm,
    reldist_diag,
    reldist_orth,
    reldist_skew,
    reldist_symm,
    row_corr,
)
from tsdm.metrics import Loss
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.viz import center_axes, kernel_heatmap, plot_spectrum, rasterize


@torch.no_grad()
def compute_metrics(
    metrics: list | dict[str, Any], *, targets: Tensor, predics: Tensor
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
    histograms: bool | int = False,
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
    writer.add_scalar(f"{identifier}:linalg/col-corr", col_corr(K), i)
    writer.add_scalar(f"{identifier}:linalg/row-corr", row_corr(K), i)

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
    histograms: bool | int = False,
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
    histograms: bool | int = False,
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
    metrics: Sequence[str] | dict[str, Loss],
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
        Callable[[nn.Module, tuple], tuple],
        Callable[[nn.Module, tuple], ResultTuple],
        Callable[[nn.Module, tuple], ResultDict],
    ]
    checkpoint_dir: Path

    history: DataFrame = field(init=False)
    logging_dir: Path = field(init=False)
    results_dir: Optional[Path] = None

    _warned_tuple: bool = False

    def __post_init__(self) -> None:
        r"""Initialize logger."""
        self.logging_dir = Path(self.writer.log_dir)
        columns = MultiIndex.from_product([self.dataloaders, self.metrics])
        index = Index([], name="epoch", dtype=int)
        self.history = DataFrame(index=index, columns=columns, dtype="Float32")

        if self.results_dir is not None:
            self.results_dir = Path(self.results_dir)

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}"

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> ResultTuple:
        r"""Get all predictions for a dataloader."""
        predics = []
        targets = []
        for batch in dataloader:
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

        return ResultTuple(targets=torch.cat(targets), predics=torch.cat(predics))

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
        histograms: bool | int = True,
        kernel_information: bool | int = 1,
        model_state: bool | int = 10,
        optimizer_state: bool | int = False,
        make_checkpoint: bool | int = 10,
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

    def log_all_metrics(self, i: int) -> None:
        r"""Log all metrics for all dataloaders."""
        if i not in self.history.index:
            empty_row = DataFrame(
                index=[i], columns=self.history.columns, dtype="Float32"
            )
            self.history = pd.concat([self.history, empty_row], sort=True)

        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            values = compute_metrics(
                self.metrics, targets=result.targets, predics=result.predics
            )

            for metric, value in values.items():
                self.history.loc[i, (key, metric)] = value.cpu().item()

            log_values(
                i,
                writer=self.writer,
                values=values,
                postfix=key,
            )

    def log_hparams(self, i: int, /) -> None:
        r"""Log hyperparameters."""
        # find best epoch on the smoothed validation curve
        best_epochs = self.history.rolling(5, center=True).mean().idxmin()

        scores = {
            split: {
                metric: float(self.history.loc[idx, (split, metric)])
                for metric, idx in best_epochs["valid"].items()
            }
            for split in ("train", "valid", "test")
        }

        if self.results_dir is not None:
            with open(self.results_dir / f"{i}.yaml", "w", encoding="utf8") as file:
                file.write(yaml.dump(scores))

        # add prefix
        test_scores = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}

        self.writer.add_hparams(
            hparam_dict=self.hparam_dict, metric_dict=test_scores, run_name="hparam"
        )
        print(f"{test_scores=} achieved by {self.hparam_dict=}")

        # FIXME: https://github.com/pytorch/pytorch/issues/32651
        for files in (self.logging_dir / "hparam").iterdir():
            shutil.move(files, self.logging_dir)
        (self.logging_dir / "hparam").rmdir()

    def log_history(self, i: int, /) -> None:
        r"""Store history dataframe to file (default format: parquet)."""
        assert self.results_dir is not None
        path = self.results_dir / f"history-{i}.parquet"
        self.history.to_parquet(path)

    def log_optimizer_state(
        self,
        i: int,
        /,
        *,
        histograms: bool | int = False,
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
        histograms: bool | int = False,
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
        histograms: bool | int = False,
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        r"""Log kernel information."""
        assert hasattr(self.model, "kernel") and isinstance(self.model.kernel, Tensor)
        log_kernel_information(
            i,
            writer=self.writer,
            kernel=self.model.kernel,
            histograms=histograms,
            prefix=prefix,
            postfix=postfix,
        )
