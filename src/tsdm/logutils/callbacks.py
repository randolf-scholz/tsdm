"""Callback utilities for logging."""

__all__ = [
    # Functions
    "compute_metrics",
    "make_checkpoint",
    "log_kernel",
    "log_metrics",
    "log_model",
    "log_optimizer",
    "log_values",
    "log_table",
    # Classes
    # Protocols
    "Callback",
    "LogFunction",
    # Callbacks
    "BaseCallback",
    "CheckpointCallback",
    "ConfigCallback",
    "EvaluationCallback",
    "HParamCallback",
    "KernelCallback",
    "LRSchedulerCallback",
    "MetricsCallback",
    "ModelCallback",
    "OptimizerCallback",
    "ScalarsCallback",
    "TableCallback",
]

import json
import logging
import pickle
from abc import ABCMeta
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    runtime_checkable,
)

import torch
import yaml
from matplotlib.pyplot import Figure
from matplotlib.pyplot import close as close_figure
from pandas import DataFrame, MultiIndex
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

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
from tsdm.metrics import LOSSES, Loss
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.types.aliases import JSON, PathLike
from tsdm.types.variables import ParameterVar as P
from tsdm.utils import mandatory_kwargs
from tsdm.utils.strings import repr_object
from tsdm.viz import center_axes, kernel_heatmap, plot_spectrum, rasterize


class TargetsAndPredics(NamedTuple):
    """Targets and predictions."""

    targets: Tensor
    predics: Tensor


class LogFunction(Protocol):
    """Protocol for logging functions."""

    def __call__(
        self,
        i: int,
        logged_object: Any,
        writer: SummaryWriter,
        /,
        *,
        name: str = "",
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        """Log to tensorboard."""


@torch.no_grad()
def compute_metrics(
    metrics: Sequence[str | Loss | type[Loss]] | Mapping[str, str | Loss | type[Loss]],
    /,
    *,
    targets: Tensor,
    predics: Tensor,
) -> dict[str, Tensor]:
    r"""Compute multiple metrics."""
    results: dict[str, Tensor] = {}

    if isinstance(metrics, Sequence):
        for metric in metrics:
            if isinstance(metric, str):
                metric = LOSSES[metric]
            if isinstance(metric, type) and callable(metric):
                results[metric.__name__] = metric()(targets, predics)
            elif callable(metric):
                results[metric.__class__.__name__] = metric(targets, predics)
            else:
                raise ValueError(f"{type(metric)=} not understood!")
        return results

    for name, metric in metrics.items():
        if isinstance(metric, str):
            metric = LOSSES[metric]
        if isinstance(metric, type) and callable(metric):
            results[name] = metric()(targets, predics)
        elif callable(metric):
            results[name] = metric(targets, predics)
        else:
            raise ValueError(f"{type(metric)=} not understood!")
    return results


def make_checkpoint(i: int, /, objects: Mapping[str, Any], path: PathLike) -> None:
    """Save checkpoints of given paths."""
    path = Path(path) / f"{i}"
    path.mkdir(parents=True, exist_ok=True)

    for name, obj in objects.items():
        match obj:
            case None:
                pass
            case torch.jit.ScriptModule():
                torch.jit.save(obj, path / name)
            case torch.nn.Module():
                torch.save(obj, path / name)
            case torch.optim.Optimizer():
                torch.save(obj, path / name)
            case LRScheduler():
                torch.save(obj, path / name)
            case dict() | list() | tuple() | set() | str() | int() | float() | None:
                with open(path / f"{name}.yaml", "w", encoding="utf8") as file:
                    yaml.safe_dump(obj, file)
            case _:
                with open(path / f"{name}.pickle", "wb") as file:
                    pickle.dump(obj, file)


def log_config(
    i: int | str,
    /,
    config: JSON,
    writer: SummaryWriter | Path,
    *,
    fmt: Literal["json", "yaml", "toml"] = "yaml",
    name: str = "config",
    prefix: str = "",
    postfix: str = "",
) -> None:
    """Log config to tensorboard."""
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"
    path = Path(writer.log_dir if isinstance(writer, SummaryWriter) else writer)
    path = path / f"{identifier}-{i}.{fmt}"
    match fmt:
        case "json":
            with open(path, "w", encoding="utf8") as file:
                json.dump(config, file)
        case "yaml":
            with open(path, "w", encoding="utf8") as file:
                yaml.safe_dump(config, file)
        case "toml":
            raise NotImplementedError("toml not implemented yet!")
        case _:
            raise ValueError(f"{fmt=} not understood!")


@torch.no_grad()
def log_kernel(
    i: int,
    /,
    kernel: Tensor,
    writer: SummaryWriter,
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
    lr_scheduler: LRScheduler,
    writer: SummaryWriter,
    *,
    name: str = "lr_scheduler",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log learning rate scheduler."""
    # identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"
    raise NotImplementedError("Not implemented yet.")


def log_metrics(
    i: int,
    /,
    metrics: Sequence[str | Loss | type[Loss]] | Mapping[str, str | Loss | type[Loss]],
    writer: SummaryWriter,
    *,
    inputs: Optional[Mapping[Literal["targets", "predics"], Tensor]] = None,
    targets: Optional[Tensor] = None,
    predics: Optional[Tensor] = None,
    key: str = "",
    name: str = "metrics",
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

    scalars = compute_metrics(metrics, targets=targets, predics=predics)
    log_values(i, scalars, writer, key=key, name=name, prefix=prefix, postfix=postfix)


@torch.no_grad()
def log_model(
    i: int,
    /,
    model: Model,
    writer: SummaryWriter,
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
        writer.add_scalar(
            f"{identifier}/gradients", multi_norm(list(variables.values())), i
        )
        writer.add_scalar(
            f"{identifier}/variables", multi_norm(list(gradients.values())), i
        )

    if log_histograms:
        for key, weight in variables.items():
            if not weight.numel():
                continue
            print(weight)
            writer.add_histogram(f"{identifier}:variables/{key}", weight, i)

        for key, gradient in gradients.items():
            if not gradient.numel():
                continue
            writer.add_histogram(f"{identifier}:gradients/{key}", gradient, i)


def log_optimizer(
    i: int,
    /,
    optimizer: Optimizer,
    writer: SummaryWriter,
    *,
    log_histograms: bool = True,
    log_norms: bool = True,
    name: str = "optimizer",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log optimizer data under ``prefix:optimizer:postfix/``."""
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"

    variables = list(optimizer.state.values())
    gradients = [w.grad for w in variables]
    moments_1 = [d["exp_avg"] for d in optimizer.state.values() if "exp_avg" in d]
    moments_2 = [d["exp_avg_sq"] for d in optimizer.state.values() if "exp_avg_sq" in d]
    log = writer.add_scalar

    if log_norms:
        log(f"{identifier}/model-gradients", multi_norm(gradients), i)
        log(f"{identifier}/model-variables", multi_norm(variables), i)
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


def log_values(
    i: int,
    /,
    scalars: Mapping[str, Tensor],
    writer: SummaryWriter,
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
    table: DataFrame,
    writer: SummaryWriter | Path,
    *,
    options: Optional[dict[str, Any]] = None,
    filetype: str = "parquet",
    name: str = "tables",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log multiple metrics at once."""
    options = {} if options is None else options
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"
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
class Callback(Protocol[P]):
    """Protocol for callbacks.

    Callbacks should be initialized with mutable objects that are being logged.
    every time the callback is called, it should log the current state of the
    mutable object.
    """

    required_kwargs: ClassVar[set[str]]
    """The required kwargs for the callback."""

    @property
    def frequency(self) -> int:
        """The frequency at which the callback is called."""

    def __call__(self, i: int, /, **kwargs: P.kwargs) -> None:
        """Log something at time index i."""


class CallbackMetaclass(ABCMeta):
    """Metaclass for callbacks."""

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


@dataclass(repr=False)
class BaseCallback(Generic[P], metaclass=CallbackMetaclass):
    """Base class for callbacks."""

    LOGGER: ClassVar[logging.Logger]
    """The debug-logger for the callback."""
    required_kwargs: ClassVar[set[str]]
    """The required kwargs for the callback."""

    _: KW_ONLY

    frequency: int = 1
    """The frequency at which the callback is called."""

    def __init_subclass__(cls) -> None:
        """Automatically set the required kwargs for the callback."""
        cls.required_kwargs = mandatory_kwargs(cls.callback)

    # @abstractmethod
    def callback(self, i: int, /, **kwargs: P.kwargs) -> None:
        """Log something at the end of a batch/epoch."""

    @wraps(callback)
    def __call__(self, i: int, /, **kwargs: P.kwargs) -> None:
        """Log something at the end of a batch/epoch."""
        if i % self.frequency == 0:
            self.callback(i, **kwargs)
        else:
            self.LOGGER.debug("Skipping callback.")

    def __repr__(self):
        """Return a string representation of the callback."""
        return repr_object(self)


@dataclass
class ConfigCallback(BaseCallback):
    """Callback to log the config to tensorboard."""

    config: JSON
    writer: SummaryWriter | Path

    _: KW_ONLY

    fmt: Literal["json", "yaml", "toml"] = "yaml"
    name: str = "config"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **_: Any) -> None:
        """Log the config to tensorboard."""
        log_config(
            i,
            config=self.config,
            writer=self.writer,
            fmt=self.fmt,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class EvaluationCallback(BaseCallback):
    """Callback to log evaluation metrics to tensorboard."""

    _: KW_ONLY

    dataloaders: Mapping[str, DataLoader]
    metrics: Sequence[str | Loss | type[Loss]] | Mapping[str, str | Loss | type[Loss]]
    model: Model
    predict_fn: Callable[..., tuple[Tensor, Tensor]]
    writer: SummaryWriter

    history: DataFrame = NotImplemented
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    best_epoch: DataFrame = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the callback."""
        if self.history is NotImplemented:
            self.history = DataFrame(
                columns=MultiIndex.from_product([self.dataloaders, self.metrics])
            )
        self.best_epoch = self.history.copy()

        for i in self.history.index:
            self.compute_results(i)

        # make sure there is exactly one validation split.
        candidate = None
        for key in self.dataloaders:
            if key.lower() in ("val", "valid", "validation"):
                if candidate is not None:
                    raise ValueError(
                        f"Found multiple validation splits: {candidate} and {key}!"
                    )
                candidate = key
        if candidate is None:
            raise ValueError("Could not find a validation split!")
        self._val_key = candidate

    def callback(self, i: int, /, **kwargs: Any) -> None:
        # update the history
        self.compute_results(i)
        self.compute_best_epoch(i)

        for key in self.dataloaders:
            log_values(
                i,
                scalars=self.history.loc[i, key].to_dict(),
                writer=self.writer,
                key=key,
                name=self.name,
                prefix=self.prefix,
                postfix=self.postfix,
            )
            log_values(
                i,
                scalars=self.best_epoch.loc[i, key].to_dict(),
                writer=self.writer,
                key=key,
                name="score",
                prefix=self.prefix,
                postfix=self.postfix,
            )

        path = Path(self.writer.log_dir) / f"history-{i}.parquet"
        self.history.to_parquet(path)

    def compute_results(self, i: int) -> None:
        """Return the results at the minimal validation loss."""
        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            scalars = compute_metrics(
                self.metrics, targets=result.targets, predics=result.predics
            )
            for metric, value in scalars.items():
                self.history.loc[i, (key, metric)] = value.cpu().item()

    def compute_best_epoch(self, i: int) -> None:
        """Compute the best epoch for the given index."""
        # unoptimized...
        mask = self.history.index <= i
        best_epochs = (
            self.history.loc[mask, self._val_key]
            .rolling(5, min_periods=1)
            .mean()
            .idxmin()
        )
        for key in self.dataloaders:
            for metric in self.metrics:
                best_value = self.history.loc[best_epochs[metric], (key, metric)]
                self.best_epoch.loc[i, (key, metric)] = best_value

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> TargetsAndPredics:
        """Return the targets and predictions for the given dataloader."""
        targets_list: list[Tensor] = []
        predics_list: list[Tensor] = []

        for batch in tqdm(dataloader, desc="Evaluation", leave=False):
            target, predic = self.predict_fn(self.model, batch)
            targets_list.append(target)
            predics_list.append(predic)

        targets = torch.nn.utils.rnn.pad_sequence(
            chain.from_iterable(targets_list),  # type: ignore[arg-type]
            batch_first=True,
            padding_value=torch.nan,
        ).squeeze()

        predics = torch.nn.utils.rnn.pad_sequence(
            chain.from_iterable(predics_list),  # type: ignore[arg-type]
            batch_first=True,
            padding_value=torch.nan,
        ).squeeze()

        return TargetsAndPredics(targets=targets, predics=predics)


# @dataclass
# class HParamCallback(BaseCallback):
#     """Callback to log hyperparameters to tensorboard."""
#
#     hparam_dict: dict[str, Any]
#     writer: SummaryWriter
#
#     _: KW_ONLY
#
#     def callback(self, i: int | str, /, **objects: Any) -> None:
#         """Log the test-error selected by validation-error."""
#
#         # add postfix
#         test_scores = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}
#
#         self.writer.add_hparams(
#             hparam_dict=self.hparam_dict, metric_dict=test_scores, run_name="hparam"
#         )
#         print(f"{test_scores=} achieved by {self.hparam_dict=}")
#
#         # # FIXME: https://github.com/pytorch/pytorch/issues/32651
#         # # os.path.dirname(os.path.realpath(__file__)) + os.sep + log_path
#         # for files in (self.log_dir / "hparam").iterdir():
#         #     shutil.move(files, self.log_dir)
#         # (self.log_dir / "hparam").rmdir()


@dataclass
class CheckpointCallback(BaseCallback):
    """Callback to save checkpoints."""

    objects: Mapping[str, object]
    path: PathLike

    _: KW_ONLY

    def callback(self, i: int, /, **objects: Any) -> None:
        make_checkpoint(i, self.objects, path=self.path)


@dataclass
class HParamCallback(BaseCallback):
    """Callback to log hyperparameters to tensorboard."""

    history: DataFrame
    hparam_dict: JSON
    results_dir: PathLike

    _: KW_ONLY

    writer: SummaryWriter

    def callback(self, i: int, /, **objects: Any) -> None:
        best_epochs = self.history.rolling(5, center=True).mean().idxmin()

        scores: dict[str, dict[str, float]] = {
            split: {
                metric: float(self.history.loc[idx, (split, metric)])
                for metric, idx in best_epochs["valid"].items()
            }
            for split in ("train", "valid", "test")
        }

        metric_dict = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}
        run_name = Path(self.writer.log_dir).absolute()

        with open(run_name / f"{i}.yaml", "w", encoding="utf8") as file:
            file.write(yaml.dump(scores))

        self.writer.add_hparams(
            hparam_dict=self.hparam_dict, metric_dict=metric_dict, run_name=run_name
        )
        print(f"Scores {metric_dict=} achieved by {self.hparam_dict=}")


@dataclass
class KernelCallback(BaseCallback):
    """Callback to log kernel information to tensorboard."""

    kernel: Tensor
    writer: SummaryWriter

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

    def callback(self, i: int, /, **state_dict: Any) -> None:
        log_kernel(
            i,
            kernel=self.kernel,
            writer=self.writer,
            log_figures=self.log_figures,
            log_scalars=self.log_scalars,
            log_distances=self.log_distances,
            log_heatmap=self.log_heatmap,
            log_linalg=self.log_linalg,
            log_norms=self.log_norms,
            log_scaled_norms=self.log_scaled_norms,
            log_spectrum=self.log_spectrum,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class LRSchedulerCallback(BaseCallback):
    """Callback to log learning rate information to tensorboard."""

    lr_scheduler: LRScheduler
    writer: SummaryWriter

    _: KW_ONLY

    name: str = "lr_scheduler"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        log_lr_scheduler(
            i,
            lr_scheduler=self.lr_scheduler,
            writer=self.writer,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class MetricsCallback(BaseCallback):
    """Callback to log multiple metrics to tensorboard."""

    metrics: Sequence[str | Loss | type[Loss]] | Mapping[str, str | Loss | type[Loss]]
    writer: SummaryWriter

    _: KW_ONLY

    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, *, targets: Tensor, predics: Tensor) -> None:  # type: ignore[override]
        log_metrics(
            i,
            metrics=self.metrics,
            writer=self.writer,
            targets=targets,
            predics=predics,
            key=self.key,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class ModelCallback(BaseCallback):
    """Callback to log model information to tensorboard."""

    model: Model
    writer: SummaryWriter

    _: KW_ONLY

    log_histograms: bool = False
    log_norms: bool = True
    name: str = "model"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        if "model" in state_dict:
            self.model = state_dict["model"]
        log_model(
            i,
            model=self.model,
            writer=self.writer,
            log_histograms=self.log_histograms,
            log_norms=self.log_norms,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class OptimizerCallback(BaseCallback):
    """Callback to log optimizer information to tensorboard."""

    optimizer: Optimizer
    writer: SummaryWriter

    _: KW_ONLY

    log_histograms: bool = False
    log_norms: bool = True
    name: str = "optimizer"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        if "optimizer" in state_dict:
            self.optimizer = state_dict["optimizer"]
        log_optimizer(
            i,
            optimizer=self.optimizer,
            writer=self.writer,
            log_histograms=self.log_histograms,
            log_norms=self.log_norms,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class ScalarsCallback(BaseCallback):
    """Callback to log multiple values to tensorboard."""

    scalars: dict[str, Tensor]
    writer: SummaryWriter

    _: KW_ONLY

    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        for key, value in state_dict.items():
            if key in self.scalars:
                self.scalars[key] = value
        log_values(
            i,
            scalars=self.scalars,
            writer=self.writer,
            key=self.key,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class TableCallback(BaseCallback):
    """Callback to log a table to disk."""

    table: DataFrame
    writer: SummaryWriter | Path

    _: KW_ONLY

    options: Optional[dict[str, Any]] = None
    filetype: str = "parquet"
    name: str = "tables"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        if hasattr(self.table, "name") and isinstance(self.table.name, str):
            if self.table.name in state_dict:
                self.table = state_dict[self.table.name]
        log_table(
            i,
            table=self.table,
            writer=self.writer,
            options=self.options,
            filetype=self.filetype,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )
