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

import pickle
import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypeAlias, TypedDict, Union

import pandas as pd
import torch
import yaml
from matplotlib.pyplot import Figure
from matplotlib.pyplot import close as close_figure
from pandas import DataFrame, Index, MultiIndex
from torch import Tensor, nn
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.encoders import BaseEncoder
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
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.viz import center_axes, kernel_heatmap, plot_spectrum, rasterize

# from tqdm.autonotebook import tqdm

Loss: TypeAlias = nn.Module | Callable[[Tensor, Tensor], Tensor]


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
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log kernel information.

    Set option to true to log every epoch.
    Set option to an integer to log whenever ``i % log_interval == 0``.
    """
    identifier = f"{prefix+':'*bool(prefix)}kernel{':'*bool(postfix)+postfix}"
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
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log optimizer data under ``prefix:optimizer:postfix/``."""
    identifier = f"{prefix+':'*bool(prefix)}optimizer{':'*bool(postfix)+postfix}"

    variables = list(optimizer.state.keys())
    gradients = [w.grad for w in variables]
    moments_1 = [d["exp_avg"] for d in optimizer.state.values() if "exp_avg" in d]
    moments_2 = [d["exp_avg_sq"] for d in optimizer.state.values() if "exp_avg_sq" in d]

    if log_scalars:
        writer.add_scalar(f"{identifier}/model-gradients", multi_norm(gradients), i)
        writer.add_scalar(f"{identifier}/model-variables", multi_norm(variables), i)
        if loss is not None:
            writer.add_scalar(f"{identifier}/loss", loss, i)
        if moments_1:
            writer.add_scalar(f"{identifier}/param-moments_1", multi_norm(moments_1), i)
        if moments_2:
            writer.add_scalar(f"{identifier}/param-moments_2", multi_norm(moments_2), i)

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
    log_scalars: bool = True,
    log_histograms: bool = True,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log model data."""
    identifier = f"{prefix+':'*bool(prefix)}model{':'*bool(postfix)+postfix}"

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


def log_values(
    i: int,
    /,
    writer: SummaryWriter,
    values: Mapping[str, Tensor],
    *,
    key: str = "",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    identifier = f"{prefix+':'*bool(prefix)}metrics{':'*bool(postfix)+postfix}"
    for metric in values:
        writer.add_scalar(f"{identifier}:{metric}/{key}", values[metric], i)


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
    model: TorchModule

    _: KW_ONLY

    checkpoint_dir: Path
    dataloaders: Mapping[str, DataLoader]
    metrics: Mapping[str, Loss]
    predict_fn: Union[
        Callable[[TorchModule, tuple], tuple],
        Callable[[TorchModule, tuple], ResultTuple],
        # Callable[[TorchModule, tuple], ResultDict],
    ]
    results_dir: Optional[Path] = None

    encoder: BaseEncoder = NotImplemented
    hparam_dict: dict[str, Any] = NotImplemented
    optimizer: TorchOptimizer = NotImplemented
    lr_scheduler: TorchLRScheduler = NotImplemented

    # extra fields
    history: DataFrame = field(init=False)
    logging_dir: Path = field(init=False)

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

    # @torch.no_grad()
    # def get_all_predictions(self, dataloader: DataLoader) -> ResultTuple:
    #     r"""Get all predictions for a dataloader."""
    #     predics = []
    #     targets = []
    #
    #     # determine the type of the output of the predict_fn
    #     iloader = iter(dataloader)
    #     batch = next(iloader)
    #     result = self.predict_fn(self.model, batch)
    #
    #     if isinstance(result, dict):
    #         targets.append(result["targets"])
    #         predics.append(result["predics"])
    #         for batch in iloader:
    #             result = self.predict_fn(self.model, batch)
    #             targets.append(result["targets"])
    #             predics.append(result["predics"])
    #     elif isinstance(result, ResultTuple):
    #         targets.append(result.targets)
    #         predics.append(result.predics)
    #         for batch in iloader:
    #             result = self.predict_fn(self.model, batch)
    #             targets.append(result.targets)
    #             predics.append(result.predics)
    #     elif isinstance(result, tuple):
    #
    #         targets.append(result[0])
    #         predics.append(result[1])
    #         for batch in iloader:
    #             result = self.predict_fn(self.model, batch)
    #             targets.append(result[0])
    #             predics.append(result[1])
    #
    #     masks = [~torch.isnan(p) for p in predics]
    #     return ResultTuple(
    #         targets=torch.cat([t[m] for t, m in zip(targets, masks)]).squeeze(),
    #         predics=torch.cat([p[m] for p, m in zip(predics, masks)]).squeeze(),
    #     )

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> ResultTuple:
        r"""Get all predictions for a dataloader."""
        # TODO: generic version
        targets = []
        predics = []

        for batch in dataloader:
            result = self.predict_fn(self.model, batch)
            targets.append(result[0])
            predics.append(result[1])

        return ResultTuple(
            # batch the list of batches by converting to list of single elements
            targets=torch.nn.utils.rnn.pad_sequence(
                chain.from_iterable(targets), batch_first=True, padding_value=torch.nan  # type: ignore[arg-type]
            ).squeeze(),
            predics=torch.nn.utils.rnn.pad_sequence(
                chain.from_iterable(predics), batch_first=True, padding_value=torch.nan  # type: ignore[arg-type]
            ).squeeze(),
        )

        # masks = [~torch.isnan(p) for p in predics]

        # return ResultTuple(
        #     targets=torch.cat([t[m] for t, m in zip(targets, masks)]).squeeze(),
        #     predics=torch.cat([p[m] for p, m in zip(predics, masks)]).squeeze(),
        # )

    def make_checkpoint(self, i: int, /) -> None:
        r"""Make checkpoint."""
        path = self.checkpoint_dir / f"{i}"
        path.mkdir(parents=True, exist_ok=True)

        # serialize the model
        if isinstance(self.model, torch.jit.ScriptModule):
            torch.jit.save(self.model, path / self.model.original_name)
        else:
            torch.save(self.model, path / self.model.__class__.__name__)

        # serialize the optimizer
        if self.optimizer is not NotImplemented:
            torch.save(self.optimizer, path / self.optimizer.__class__.__name__)

        # serialize the lr scheduler
        if self.lr_scheduler is not NotImplemented:
            torch.save(self.lr_scheduler, path / self.lr_scheduler.__class__.__name__)

        # serialize the hyperparameters
        if self.hparam_dict is not NotImplemented:
            with open(path / "hparams.yaml", "w", encoding="utf8") as file:
                yaml.safe_dump(self.hparam_dict, file)

        # serialize the hyperparameters
        if self.encoder is not NotImplemented:
            with open(path / "encoder.pickle", "wb") as file:
                pickle.dump(self.encoder, file)

    def log_metrics(
        self,
        i: int,
        /,
        *,
        targets: Tensor,
        predics: Tensor,
        key: str = "",
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
            key=key,
            prefix=prefix,
            postfix=postfix,
        )

    def log_batch_end(
        self,
        i: int,
        *,
        targets: Optional[Tensor] = None,
        predics: Optional[Tensor] = None,
        loss: Optional[Tensor] = None,
    ) -> None:
        r"""Log batch end."""
        if targets is not None and predics is not None:
            self.log_metrics(i, targets=targets, predics=predics, key="batch")
        self.log_optimizer_state(i, loss=loss, log_scalars=True, log_histograms=False)

    def log_epoch_end(
        self,
        i: int,
        *,
        # individual switches
        log_kernel: bool | int = True,
        log_model: bool | int = True,
        log_optimizer: bool | int = False,
        make_checkpoint: bool | int = 10,
    ) -> None:
        r"""Log epoch end."""
        self.log_all_metrics(i)

        if log_kernel and i % log_kernel == 0:
            self.log_kernel_information(i)

        if log_model and i % log_model == 0:
            self.log_model_state(i, log_scalars=False, log_histograms=True)

        if log_optimizer and i % log_optimizer == 0:
            self.log_optimizer_state(i, log_scalars=False, log_histograms=True)

        if make_checkpoint and i % make_checkpoint == 0:
            self.make_checkpoint(i)

    def log_all_metrics(self, i: int) -> None:
        r"""Log all metrics for all dataloaders."""
        if i not in self.history.index:
            empty_row = DataFrame(
                index=[i], columns=self.history.columns, dtype="Float32"
            )
            self.history = pd.concat([self.history, empty_row], sort=True)

        if not self._warned_tuple:
            self._warned_tuple = True
            warnings.warn(
                "We recommend using a NamedTuple or a dictionary as the return type."
                " Ensure that 'targets'/'predics' is the first/second output.",
                UserWarning,
            )

        # Schema: identifier:category/key, e.g. `metrics:MSE/train`
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
                key=key,
            )

    def log_hparams(self, i: int, /) -> None:
        r"""Log hyperparameters."""
        # Find the best epoch on the smoothed validation curve
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

        # add postfix
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

    def log_optimizer_state(self, i: int, /, **kwds: Any) -> None:
        r"""Log optimizer state."""
        assert self.optimizer is not NotImplemented
        log_optimizer_state(i, writer=self.writer, optimizer=self.optimizer, **kwds)

    def log_model_state(self, i: int, /, **kwds: Any) -> None:
        r"""Log model state."""
        assert self.model is not NotImplemented
        log_model_state(i, writer=self.writer, model=self.model, **kwds)

    def log_kernel_information(self, i: int, /, **kwds: Any) -> None:
        r"""Log kernel information."""
        assert self.model is not NotImplemented
        assert hasattr(self.model, "kernel") and isinstance(self.model.kernel, Tensor)
        log_kernel_information(i, writer=self.writer, kernel=self.model.kernel, **kwds)
