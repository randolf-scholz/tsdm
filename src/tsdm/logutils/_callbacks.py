"""Callback utilities for logging."""

__all__ = [
    # Functions
    "compute_metrics",
    "make_checkpoint",
    "log_kernel",
    "log_metrics",
    "log_model",
    "log_optimizer",
    "log_scalars",
    "log_table",
    # Classes
    # Protocols
    "Callback",
    "LogFunction",
    # Callbacks
    "BaseCallback",
    "CheckpointCallback",
    "KernelCallback",
    "LRSchedulerCallback",
    "MetricsCallback",
    "ModelCallback",
    "OptimizerCallback",
    "ScalarsCallback",
    "TableCallback",
]

import logging
import pickle
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass
from pathlib import Path

# from tsdm.types.variables import ParameterVar as P
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import torch
import yaml
from matplotlib.pyplot import Figure
from matplotlib.pyplot import close as close_figure
from pandas import DataFrame
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.tensorboard import SummaryWriter

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
from tsdm.viz import center_axes, kernel_heatmap, plot_spectrum, rasterize

# class LogFunction(Protocol):
#     """Protocol for logging functions."""
#
#     def __call__(
#         self,
#         i: int,
#         /,
#         writer: SummaryWriter,
#         name: str = "optimizer",
#         prefix: str = "",
#         postfix: str = "",
#     ) -> None:
#         """Log to tensorboard."""

P = ParamSpec("P")
# https://youtrack.jetbrains.com/issue/PY-59329/
_LogFunction: TypeAlias = Callable[Concatenate[int, P], None]
LogFunction: TypeAlias = _LogFunction[...]  # type: ignore[misc]


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


def make_checkpoint(i: int, /, path: Path, **objects: Any) -> None:
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


@torch.no_grad()
def log_kernel(
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


def log_lr_scheduler(
    i: int,
    /,
    writer: SummaryWriter,
    lr_scheduler: LRScheduler,
    *,
    name: str = "lr_scheduler",
    prefix: str = "",
    postfix: str = "",
) -> None:
    # identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"
    raise NotImplementedError("Not implemented yet.")


def log_metrics(
    i: int,
    /,
    writer: SummaryWriter,
    metrics: Sequence[str] | Mapping[str, Loss],
    *,
    inputs: Optional[Mapping[Literal["targets", "predics"], Tensor]] = None,
    targets: Optional[Tensor] = None,
    predics: Optional[Tensor] = None,
    key: str = "",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    if targets is not None and predics is not None and inputs is None:
        pass
    elif targets is None and predics is None and inputs is not None:
        targets = inputs["targets"]
        predics = inputs["predics"]
    else:
        raise ValueError("Either `inputs` or `targets` and `predics` must be provided.")

    assert len(targets) == len(predics)
    assert isinstance(metrics, Mapping)

    values = compute_metrics(metrics, targets=targets, predics=predics)
    log_scalars(i, writer, values, key=key, prefix=prefix, postfix=postfix)


@torch.no_grad()
def log_model(
    i: int,
    /,
    writer: SummaryWriter,
    model: Model,
    *,
    log_histograms: bool = True,
    log_norms: bool = True,
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

    if log_norms:
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


def log_optimizer(
    i: int,
    /,
    writer: SummaryWriter,
    optimizer: Optimizer,
    *,
    log_histograms: bool = True,
    log_norms: bool = True,
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

    if log_norms:
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


def log_scalars(
    i: int,
    /,
    writer: SummaryWriter,
    scalars: Mapping[str, Tensor],
    *,
    key: str = "",
    name: str = "metrics",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"
    for _id, scalar in scalars.items():
        writer.add_scalar(f"{identifier}:{_id}/{key}", scalar, i)


def log_table(
    i: int,
    /,
    writer: SummaryWriter | Path,
    table: DataFrame,
    *,
    options: Optional[dict[str, Any]] = None,
    filetype: str = "parquet",
    key: str = "",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    options = {} if options is None else options
    identifier = f"{prefix+':'*bool(prefix)}{key}{':'*bool(postfix)+postfix}"
    path = Path(writer.log_dir if isinstance(writer, SummaryWriter) else writer)
    path = path / f"{identifier+'-'*bool(identifier)}{i}"

    match filetype:
        case "parquet":
            table.to_parquet(f"{path}.parquet", **options)
        case "csv":
            table.to_csv(f"{path}.csv", **options)
        case "feather":
            table.to_feather(f"{path}.feather", **options)
        case "json":
            table.to_json(f"{path}.json", **options)
        case "pickle":
            table.to_pickle(f"{path}.pickle", **options)
        case _:
            raise ValueError(f"Unknown {filetype=!r}!")


@runtime_checkable
class Callback(Protocol):
    """Protocol for callbacks.

    Callbacks should be initialized with mutable objects that are being logged.
    every time the callback is called, it should log the current state of the
    mutable object.
    """

    # @property
    # def log_dir(self) -> Path:
    #     """The Path where things will be logged in."""

    @property
    def frequency(self) -> int:
        """The frequency at which the callback is called."""

    def __call__(self, i: int, /, **state_dict: Any) -> None:
        """Log something at time index i."""


class CallbackMetaclass(ABCMeta):
    """Metaclass for callbacks."""

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


@dataclass
class BaseCallback(metaclass=CallbackMetaclass):
    """Base class for callbacks."""

    LOGGER: ClassVar[logging.Logger]
    """The debug-logger for the callback."""

    _: KW_ONLY

    frequency: int = 1
    """The frequency at which the callback is called."""

    @abstractmethod
    def __callback(self, i: int, /, **state_dict: Any) -> None:
        """Log something at the end of a batch/epoch."""

    def __call__(self, i: int, /, **state_dict: Any) -> None:
        """Log something at the end of a batch/epoch."""
        if i % self.frequency == 0:
            self.__callback(i, **state_dict)
        else:
            self.LOGGER.debug("Skipping callback.")


@dataclass
class CheckpointCallback(BaseCallback):
    """Callback to save checkpoints."""

    path: Path
    objects: dict[str, object]
    _: KW_ONLY

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        for key in state_dict:
            if key in self.objects:
                self.objects[key] = state_dict.pop(key)
        make_checkpoint(i, self.path, **self.objects)


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

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        log_kernel(i, **asdict(self))


@dataclass
class LRSchedulerCallback(BaseCallback):
    """Callback to log learning rate information to tensorboard."""

    writer: SummaryWriter
    scheduler: LRScheduler
    _: KW_ONLY
    key: str = ""
    prefix: str = ""
    postfix: str = ""

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        log_lr_scheduler(i, **asdict(self))


@dataclass
class MetricsCallback(BaseCallback):
    """Callback to log multiple metrics to tensorboard."""

    writer: SummaryWriter
    metrics: Sequence[str] | Mapping[str, Loss]
    _: KW_ONLY
    key: str = ""
    prefix: str = ""
    postfix: str = ""

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        if "targets" not in state_dict and "predics" not in state_dict:
            raise RuntimeWarning("No targets or predictions found in state_dict!")
        targets = state_dict.pop("targets")
        predics = state_dict.pop("predics")
        log_metrics(i, **asdict(self), targets=targets, predics=predics)


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

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        if "model" in state_dict:
            self.model = state_dict.pop("model")
        log_model(i, **asdict(self))


@dataclass
class OptimizerCallback(BaseCallback):
    """Callback to log optimizer information to tensorboard."""

    writer: SummaryWriter
    optimizer: Optimizer
    _: KW_ONLY
    log_histograms: bool = True
    log_scalars: bool = True
    loss: Optional[float | Tensor] = None
    name: str = "optimizer"
    prefix: str = ""
    postfix: str = ""

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        if "optimizer" in state_dict:
            self.optimizer = state_dict.pop("optimizer")
        log_optimizer(i, **asdict(self))


@dataclass
class ScalarsCallback(BaseCallback):
    """Callback to log multiple values to tensorboard."""

    writer: SummaryWriter
    scalars: dict[str, Tensor]
    _: KW_ONLY
    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        for key in state_dict:
            if key in self.scalars:
                self.scalars[key] = state_dict.pop(key)
        log_scalars(i, **asdict(self))


@dataclass
class TableCallback(BaseCallback):
    """Callback to log a table to disk."""

    path: Path
    table: DataFrame
    _: KW_ONLY
    options: Optional[dict[str, Any]] = None
    filetype: str = "parquet"
    key: str = ""
    prefix: str = ""
    postfix: str = ""

    def __callback(self, i: int, /, **state_dict: Any) -> None:
        if hasattr(self.table, "name") and isinstance(self.table.name, str):
            if self.table.name in state_dict:
                self.table = state_dict.pop(self.table.name)
        log_table(i, **asdict(self))
