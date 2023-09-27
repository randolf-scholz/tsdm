r"""TODO: Add module summary line.

TODO: Add module description.
"""

__all__ = [
    # Protocols
    "LogFunction",
    # Functions
    "make_checkpoint",
    "log_config",
    "log_kernel",
    "log_lr_scheduler",
    "log_metrics",
    "log_model",
    "log_optimizer",
    "log_values",
    "log_table",
]

import inspect
import json
import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypeAlias,
    TypeGuard,
    runtime_checkable,
)

import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.pyplot import close as close_figure
from pandas import DataFrame
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from tsdm.constants import EMPTY_MAP
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
from tsdm.metrics import LOSSES, Metric
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.types.aliases import JSON, PathLike
from tsdm.types.variables import any_co as T_co, any_var as T
from tsdm.viz import center_axes, kernel_heatmap, plot_spectrum, rasterize

MaybeWrapped: TypeAlias = T_co | Callable[[], T_co] | Callable[[int], T_co]
"""Type Alias for maybe wrapped values."""


def unpack_maybewrapped(x: MaybeWrapped[T], /, *, step: int) -> T:
    r"""Unpack wrapped values."""
    if callable(x):
        try:
            return x(step)  # type: ignore[call-arg]
        except TypeError:
            return x()  # type: ignore[call-arg]

    return x


@runtime_checkable
class LogFunction(Protocol):
    """Protocol for logging functions."""

    def __call__(
        self,
        step: int,
        logged_object: Any,
        writer: SummaryWriter,
        /,
        *,
        name: str = "",
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        """Log to tensorboard."""
        ...


def is_logfunc(func: Callable, /) -> TypeGuard[LogFunction]:
    """Check if the function is a callback."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    P = inspect.Parameter
    return len(params) >= 6 and all(
        p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD) for p in params[:3]
    )


class TargetsAndPredics(NamedTuple):
    """Targets and predictions."""

    targets: Tensor
    predics: Tensor


@torch.no_grad()
def compute_metrics(
    metrics: Sequence[str | Metric | type[Metric]]
    | Mapping[str, str | Metric | type[Metric]],
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


def make_checkpoint(step: int, objects: Mapping[str, Any], path: PathLike) -> None:
    """Save checkpoints of given paths."""
    path = Path(path) / f"{step}"
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
    step: int | str,
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
    path = path / f"{identifier}-{step}.{fmt}"
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
    step: int,
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
        log(f"{identifier}:distances/diagonal_(relative)",       reldist_diag(K), step)
        log(f"{identifier}:distances/skew-symmetric_(relative)", reldist_skew(K), step)
        log(f"{identifier}:distances/symmetric_(relative)",      reldist_symm(K), step)
        log(f"{identifier}:distances/orthogonal_(relative)",     reldist_orth(K), step)
        # fmt: on

    if log_scalars and log_linalg:
        # fmt: off
        # general properties
        log(f"{identifier}:linalg/condition-number", torch.linalg.cond(K), step)
        log(f"{identifier}:linalg/correlation-cols", col_corr(K), step)
        log(f"{identifier}:linalg/correlation-rows", row_corr(K), step)
        log(f"{identifier}:linalg/effective-rank",   erank(K), step)
        log(f"{identifier}:linalg/logdet",           torch.linalg.slogdet(K)[-1], step)
        log(f"{identifier}:linalg/mean",             torch.mean(K), step)
        log(f"{identifier}:linalg/stdev",            torch.std(K), step)
        log(f"{identifier}:linalg/trace",            torch.trace(K), step)
        # fmt: on

    if log_scalars and log_norms:
        # fmt: off
        # Matrix norms
        log(f"{identifier}:norms/matrix_+∞_(max_absolut_value)", matrix_norm(K, p=+ω), step)
        log(f"{identifier}:norms/matrix_+2_(sum_squared_value)", matrix_norm(K, p=+2), step)
        log(f"{identifier}:norms/matrix_+1_(sum_absolut_value)", matrix_norm(K, p=+1), step)
        log(f"{identifier}:norms/matrix_-1_(sum_inverse_value)", matrix_norm(K, p=-1), step)
        log(f"{identifier}:norms/matrix_-2_(sum_isquare_value)", matrix_norm(K, p=-2), step)
        log(f"{identifier}:norms/matrix_-∞_(min_absolut_value)", matrix_norm(K, p=-ω), step)
        # Logarithmic norms
        log(f"{identifier}:norms/logarithmic_+∞_(max_row_sum)", logarithmic_norm(K, p=+ω), step)
        log(f"{identifier}:norms/logarithmic_+2_(max_eig_val)", logarithmic_norm(K, p=+2), step)
        log(f"{identifier}:norms/logarithmic_+1_(max_col_sum)", logarithmic_norm(K, p=+1), step)
        log(f"{identifier}:norms/logarithmic_-1_(min_col_sum)", logarithmic_norm(K, p=-1), step)
        log(f"{identifier}:norms/logarithmic_-2_(min_eig_val)", logarithmic_norm(K, p=-2), step)
        log(f"{identifier}:norms/logarithmic_-∞_(min_row_sum)", logarithmic_norm(K, p=-ω), step)
        # Operator norms
        log(f"{identifier}:norms/operator_+∞_(max_row_sum)", operator_norm(K, p=+ω), step)
        log(f"{identifier}:norms/operator_+2_(max_σ_value)", operator_norm(K, p=+2), step)
        log(f"{identifier}:norms/operator_+1_(max_col_sum)", operator_norm(K, p=+1), step)
        log(f"{identifier}:norms/operator_-1_(min_col_sum)", operator_norm(K, p=-1), step)
        log(f"{identifier}:norms/operator_-2_(min_σ_value)", operator_norm(K, p=-2), step)
        log(f"{identifier}:norms/operator_-∞_(min_row_sum)", operator_norm(K, p=-ω), step)
        # Schatten norms
        log(f"{identifier}:norms/schatten_+∞_(max_absolut_σ)", schatten_norm(K, p=+ω), step)
        log(f"{identifier}:norms/schatten_+2_(sum_squared_σ)", schatten_norm(K, p=+2), step)
        log(f"{identifier}:norms/schatten_+1_(sum_absolut_σ)", schatten_norm(K, p=+1), step)
        log(f"{identifier}:norms/schatten_-1_(sum_inverse_σ)", schatten_norm(K, p=-1), step)
        log(f"{identifier}:norms/schatten_-2_(sum_isquare_σ)", schatten_norm(K, p=-2), step)
        log(f"{identifier}:norms/schatten_-∞_(min_absolut_σ)", schatten_norm(K, p=-ω), step)
        # fmt: on

    if log_scalars and log_scaled_norms:
        # fmt: off
        # Matrix norms
        log(f"{identifier}:norms-scaled/matrix_+∞_(max_absolut_value)", matrix_norm(K, p=+ω, scaled=True), step)
        log(f"{identifier}:norms-scaled/matrix_+2_(sum_squared_value)", matrix_norm(K, p=+2, scaled=True), step)
        log(f"{identifier}:norms-scaled/matrix_+1_(sum_absolut_value)", matrix_norm(K, p=+1, scaled=True), step)
        log(f"{identifier}:norms-scaled/matrix_-1_(sum_inverse_value)", matrix_norm(K, p=-1, scaled=True), step)
        log(f"{identifier}:norms-scaled/matrix_-2_(sum_isquare_value)", matrix_norm(K, p=-2, scaled=True), step)
        log(f"{identifier}:norms-scaled/matrix_-∞_(min_absolut_value)", matrix_norm(K, p=-ω, scaled=True), step)
        # Logarithmic norms
        # log(f"{identifier}:norms-scaled/logarithmic_+∞_(max_row_sum)", logarithmic_norm(K, p=+ω, scaled=True), step)
        # log(f"{identifier}:norms-scaled/logarithmic_+2_(max_eig_val)", logarithmic_norm(K, p=+2, scaled=True), step)
        # log(f"{identifier}:norms-scaled/logarithmic_+1_(max_col_sum)", logarithmic_norm(K, p=+1, scaled=True), step)
        # log(f"{identifier}:norms-scaled/logarithmic_-1_(min_col_sum)", logarithmic_norm(K, p=-1, scaled=True), step)
        # log(f"{identifier}:norms-scaled/logarithmic_-2_(min_eig_val)", logarithmic_norm(K, p=-2, scaled=True), step)
        # log(f"{identifier}:norms-scaled/logarithmic_-∞_(min_row_sum)", logarithmic_norm(K, p=-ω, scaled=True), step)
        # Operator norms
        log(f"{identifier}:norms-scaled/operator_+∞_(max_row_sum)", operator_norm(K, p=+ω, scaled=True), step)
        log(f"{identifier}:norms-scaled/operator_+2_(max_σ_value)", operator_norm(K, p=+2, scaled=True), step)
        log(f"{identifier}:norms-scaled/operator_+1_(max_col_sum)", operator_norm(K, p=+1, scaled=True), step)
        log(f"{identifier}:norms-scaled/operator_-1_(min_col_sum)", operator_norm(K, p=-1, scaled=True), step)
        log(f"{identifier}:norms-scaled/operator_-2_(min_σ_value)", operator_norm(K, p=-2, scaled=True), step)
        log(f"{identifier}:norms-scaled/operator_-∞_(min_row_sum)", operator_norm(K, p=-ω, scaled=True), step)
        # Schatten norms
        # log(f"{identifier}:norms-scaled/schatten_+∞_(max_absolut_σ)", schatten_norm(K, p=+ω, scaled=True), step)
        # log(f"{identifier}:norms-scaled/schatten_+2_(sum_squared_σ)", schatten_norm(K, p=+2, scaled=True), step)
        # log(f"{identifier}:norms-scaled/schatten_+1_(sum_absolut_σ)", schatten_norm(K, p=+1, scaled=True), step)
        # log(f"{identifier}:norms-scaled/schatten_-1_(sum_inverse_σ)", schatten_norm(K, p=-1, scaled=True), step)
        # log(f"{identifier}:norms-scaled/schatten_-2_(sum_isquare_σ)", schatten_norm(K, p=-2, scaled=True), step)
        # log(f"{identifier}:norms-scaled/schatten_-∞_(min_absolut_σ)", schatten_norm(K, p=-ω, scaled=True), step)
        # fmt: on

    # 2d-data is order of magnitude more expensive.
    if log_figures and log_heatmap:
        writer.add_image(f"{identifier}:heatmap", kernel_heatmap(K, fmt="CHW"), step)

    if log_figures and log_spectrum:
        spectrum: Figure = plot_spectrum(K)
        spectrum = center_axes(spectrum)
        image = rasterize(spectrum, w=2, h=2, px=512, py=512)
        close_figure(spectrum)  # Important!
        writer.add_image(f"{identifier}:spectrum", image, step, dataformats="HWC")


def log_lr_scheduler(
    step: int,
    lr_scheduler: LRScheduler,
    writer: SummaryWriter,
    *,
    name: str = "lr_scheduler",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log learning rate scheduler."""
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"
    cls_name = lr_scheduler.__class__.__name__
    lr = lr_scheduler.get_last_lr()
    writer.add_scalar(f"{identifier}:{cls_name}/lr", lr, step)


def log_metrics(
    step: int,
    metrics: Sequence[str | Metric | type[Metric]]
    | Mapping[str, str | Metric | type[Metric]],
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
    log_values(
        step, scalars, writer, key=key, name=name, prefix=prefix, postfix=postfix
    )


@torch.no_grad()
def log_model(
    step: int,
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
            f"{identifier}/gradients", multi_norm(list(variables.values())), step
        )
        writer.add_scalar(
            f"{identifier}/variables", multi_norm(list(gradients.values())), step
        )

    if log_histograms:
        for key, weight in variables.items():
            if not weight.numel():
                continue
            print(weight)
            writer.add_histogram(f"{identifier}:variables/{key}", weight, step)

        for key, grad in gradients.items():
            if not grad.numel():
                continue
            writer.add_histogram(f"{identifier}:gradients/{key}", grad, step)


def log_optimizer(
    step: int,
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
        log(f"{identifier}/model-gradients", multi_norm(gradients), step)
        log(f"{identifier}/model-variables", multi_norm(variables), step)
        if moments_1:
            log(f"{identifier}/param-moments_1", multi_norm(moments_1), step)
        if moments_2:
            log(f"{identifier}/param-moments_2", multi_norm(moments_2), step)

    if log_histograms:
        for j, (w, g) in enumerate(zip(variables, gradients)):
            if not w.numel():
                continue
            writer.add_histogram(f"{identifier}:model-variables/{j}", w, step)
            writer.add_histogram(f"{identifier}:model-gradients/{j}", g, step)

        for j, (a, b) in enumerate(zip(moments_1, moments_2)):
            if not a.numel():
                continue
            writer.add_histogram(f"{identifier}:param-moments_1/{j}", a, step)
            writer.add_histogram(f"{identifier}:param-moments_2/{j}", b, step)


def log_values(
    step: int,
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
        writer.add_scalar(f"{identifier}:{_id}/{key}", scalar, step)


def log_table(
    step: int,
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
    path = path / f"{identifier+'-'*bool(identifier)}{step}"

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


def log_plot(
    step: int,
    plot: MaybeWrapped[Figure],
    writer: SummaryWriter,
    *,
    name: str = "forecastplot",
    prefix: str = "",
    postfix: str = "",
    rasterization_options: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    r"""Make a forecast plot.

    Args:
        step: The current step.
        plot: The plot to log, can be a figure, or a callable returning a figure.
        writer: The writer to log to.
        name: The name of the plot.
        prefix: The prefix of the plot.
        postfix: The postfix of the plot.
        rasterization_options: Options to pass to `tsdm.viz.rasterize`.
    """
    identifier = f"{prefix+':'*bool(prefix)}{name}{':'*bool(postfix)+postfix}"

    # generate the figure
    fig = unpack_maybewrapped(plot, step=step)

    # rasterize the figure
    image = rasterize(fig, **rasterization_options)

    writer.add_image(f"{identifier}", image, step, dataformats="HWC")
