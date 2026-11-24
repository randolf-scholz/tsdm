r"""Log to TensorBoard or save artifacts in a writer's logging directory."""

__all__ = [
    # Protocols
    "LogFunction",
    # utilities
    "is_logfunc",
    # Functions
    "log_config",
    "log_kernel",
    "log_lr_scheduler",
    "log_metrics",
    "log_model",
    "log_optimizer",
    "log_plot",
    "log_table",
    "log_values",
]

import inspect
import json
import math
from collections.abc import Callable, Mapping
from inspect import Parameter
from pathlib import Path
from typing import Any, Literal, Protocol, TypeIs

import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.pyplot import close as close_figure
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as TorchLRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.constants import EMPTY_MAP
from tsdm.linalg import (
    col_corr,
    erank,
    logarithmic_norm,
    matrix_norm,
    multi_norm,
    operator_norm,
    reldist_diagonal,
    reldist_orthogonal,
    reldist_skew,
    reldist_symmetric,
    row_corr,
    schatten_norm,
)
from tsdm.plotting import center_axes, kernel_heatmap, plot_spectrum, rasterize
from tsdm.types import FilePath
from tsdm.types.nested import JSON
from tsdm.utils import transpose_list_of_dicts

type Loss = Callable[..., Tensor]


def _eval_loss(loss: Loss, *, targets: Tensor, predictions: Tensor) -> float:
    try:
        return float(loss(targets=targets, predictions=predictions).cpu().item())  # type: ignore
    except Exception:
        # use predictions first (as in `nn.functional.cross_entropy`)
        return float(loss(predictions, targets).cpu().item())


class LogFunction(Protocol):
    r"""Protocol for logging functions."""

    def __call__(
        self,
        step: int,
        writer: SummaryWriter,
        logged_object: Any,
        /,
        *,
        name: str = "",
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        r"""Log the supplied object to TensorBoard at `step`."""
        ...


def is_logfunc(func: object, /) -> TypeIs[LogFunction]:
    r"""Check whether a callable has a signature compatible with a logging function.

    Require at least six parameters, with the first three accepting positional
    arguments. This is a signature heuristic; parameter names, annotations,
    and the remaining parameter kinds are not checked. Signature inspection
    errors propagate to the caller.
    """
    if not callable(func):
        return False

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    return len(params) >= 6 and all(
        p.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
        for p in params[:3]
    )


def log_config(
    step: int | str,
    writer: SummaryWriter | FilePath,
    /,
    config: JSON,
    *,
    fmt: Literal["json", "yaml", "toml"] = "yaml",
    name: str = "config",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Save a configuration as a JSON or YAML file in the logging directory.

    Use `writer.log_dir` when given a `SummaryWriter`, or the supplied path
    otherwise. The destination directory must already exist. The filename is
    `{identifier}-{step}.{fmt}`, where optional `prefix` and `postfix`
    surround `name`, separated by colons. TOML output raises
    `NotImplementedError`.
    """
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"
    path = Path(writer.log_dir if isinstance(writer, SummaryWriter) else writer)
    path /= f"{identifier}-{step}.{fmt}"

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
    writer: SummaryWriter,
    /,
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
    r"""Log scalar diagnostics and figures for a square, two-dimensional kernel.

    The boolean `log_scalars` and `log_figures` switches enable scalar and
    image logging, respectively. Individual switches select distances, linear
    algebra diagnostics, norms, scaled norms, a heatmap, and an eigenvalue plot.
    Each invocation logs at the supplied `step`; scheduling is handled by the
    caller or callback. A nonsquare or nonmatrix kernel raises `ValueError`.
    """
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"
    K = kernel
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("Kernel must be a square matrix!")

    log = writer.add_scalar
    ω = math.inf

    if log_scalars and log_distances:
        # fmt: off
        # relative distance to some matrix groups
        log(f"{identifier}:distances/diagonal_(relative)",       reldist_diagonal(K)  , step)
        log(f"{identifier}:distances/skew-symmetric_(relative)", reldist_skew(K)      , step)
        log(f"{identifier}:distances/symmetric_(relative)",      reldist_symmetric(K) , step)
        log(f"{identifier}:distances/orthogonal_(relative)",     reldist_orthogonal(K), step)
        # fmt: on

    if log_scalars and log_linalg:
        # fmt: off
        # general properties
        log(f"{identifier}:linalg/condition-number", torch.linalg.cond(K)       , step)
        log(f"{identifier}:linalg/correlation-cols", col_corr(K)                , step)
        log(f"{identifier}:linalg/correlation-rows", row_corr(K)                , step)
        log(f"{identifier}:linalg/effective-rank",   erank(K)                   , step)
        log(f"{identifier}:linalg/logdet",           torch.linalg.slogdet(K)[-1], step)
        log(f"{identifier}:linalg/mean",             torch.mean(K)              , step)
        log(f"{identifier}:linalg/stdev",            torch.std(K)               , step)
        log(f"{identifier}:linalg/trace",            torch.trace(K)             , step)
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
    writer: SummaryWriter,
    /,
    lr_scheduler: TorchLRScheduler,
    *,
    name: str = "lr_scheduler",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log the learning rate returned by the scheduler's `get_last_lr` method."""
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"
    cls_name = lr_scheduler.__class__.__name__
    lr = lr_scheduler.get_last_lr()
    writer.add_scalar(f"{identifier}:{cls_name}/lr", lr, step)


@torch.no_grad()
def log_metrics(
    step: int,
    writer: SummaryWriter,
    /,
    metrics: Mapping[str, Callable[[Tensor, Tensor], Tensor]],
    *,
    inputs: Mapping[Literal["predictions", "targets"], Tensor] | None = None,
    predictions: Tensor | None = None,
    targets: Tensor | None = None,
    key: str = "",
    name: str = "metrics",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Evaluate metrics on targets and predictions and log the resulting scalars.

    Supply either `inputs` with `predictions` and `targets` keys, or both
    `predictions` and `targets` directly. These forms are mutually exclusive.
    Targets and predictions must have the same length; invalid combinations or
    unequal lengths raise `ValueError`.
    """
    if predictions is not None and targets is not None and inputs is None:
        pass
    elif predictions is None and targets is None and inputs is not None:
        predictions = inputs["predictions"]
        targets = inputs["targets"]
    else:
        raise ValueError(
            "Either `inputs` or `predictions` and `targets` must be provided."
        )

    if len(predictions) != len(targets):
        raise ValueError("Targets and predictions must have the same length!")

    scalars = {
        key: _eval_loss(metric, targets=targets, predictions=predictions)
        for key, metric in metrics.items()
    }
    log_values(
        step,
        writer,
        scalars=scalars,
        key=key,
        name=name,
        prefix=prefix,
        postfix=postfix,
    )


@torch.no_grad()
def log_model(
    step: int,
    writer: SummaryWriter,
    /,
    model: nn.Module,
    *,
    log_histograms: bool = True,
    log_norms: bool = True,
    name: str = "model",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log aggregate norms and histograms of model parameters and their gradients.

    Parameters without gradients are omitted from gradient logging. Empty tensors
    are skipped when logging histograms. Use `log_norms` and `log_histograms`
    to select the outputs.
    """
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"

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
            writer.add_histogram(f"{identifier}:variables/{key}", weight, step)

        for key, grad in gradients.items():
            if not grad.numel():
                continue
            writer.add_histogram(f"{identifier}:gradients/{key}", grad, step)


def log_optimizer(
    step: int,
    writer: SummaryWriter,
    /,
    optimizer: Optimizer,
    *,
    log_histograms: bool = True,
    log_norms: bool = True,
    log_extras: bool = True,
    name: str = "optimizer",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log optimizer parameters, gradients, and additional state tensors.

    In tags, optional `prefix` and `postfix` surround `name`, separated by
    colons. The `log_norms`, `log_histograms`, and `log_extras` switches
    select aggregate norms, histograms, and additional state such as momentum.
    """
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"

    # NOTE: optimizer.state is of kind
    #  dict[tensor, {step: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor}}]
    optim_state: dict[Tensor, dict[str, Tensor]] = optimizer.state

    # get the variables and gradients
    variables = list(optim_state)
    gradients = [w.grad for w in optim_state if w.grad is not None]
    log = writer.add_scalar

    if log_norms:
        log(f"{identifier}/model-gradients", multi_norm(gradients), step)
        log(f"{identifier}/model-variables", multi_norm(variables), step)

    if log_histograms:
        for j, (w, g) in enumerate(zip(variables, gradients, strict=True)):
            if not w.numel():
                continue
            writer.add_histogram(f"{identifier}:model-variables/{j}", w, step)
            writer.add_histogram(f"{identifier}:model-gradients/{j}", g, step)

    if log_extras:
        extras: dict[str, list[Tensor]] = transpose_list_of_dicts(optim_state.values())
        extras.pop("step", None)

        if log_norms:
            for key, tensor_list in extras.items():
                log(f"{identifier}/{key}", multi_norm(tensor_list), step)

        if log_histograms:
            for key, tensor_list in extras.items():
                for j, tensor in enumerate(tensor_list):
                    if not tensor.numel():
                        continue
                    writer.add_histogram(f"{identifier}:{key}/{j}", tensor, step)


def log_values(
    step: int,
    writer: SummaryWriter,
    /,
    scalars: Mapping[str, Tensor | float],
    *,
    key: str = "",
    name: str = "metrics",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Log a mapping of precomputed scalar values to TensorBoard.

    Values may be Python numbers or scalar tensors.
    Each entry is logged at `step` under `{identifier}:{entry_name}/{key}`.
    In the identifier, optional `prefix` and `postfix` surround `name`,
    separated by colons.
    """
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"
    for ident, scalar in scalars.items():
        writer.add_scalar(f"{identifier}:{ident}/{key}", scalar, step)


def log_table[T](
    step: int,
    writer: SummaryWriter | FilePath,
    /,
    table: T,
    *,
    serializer: Callable[[T, Path], object],
    name: str,
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Serialize a table to a file in the logging directory.

    Use `writer.log_dir` when given a `SummaryWriter`, or the supplied path
    otherwise. The destination directory must already exist. The filename is
    `{identifier}-{step}{suffix}`, with the hyphen omitted for an empty
    identifier. The suffix is taken from `name`, so `name="metrics.csv"`
    produces `metrics-3.csv` at step 3. Names without a suffix produce
    extensionless files. Optional `prefix` and `postfix` surround the name's
    stem, separated by colons.

    Call `serializer(table, path)` to write the file; its return value is
    ignored. Configure serialization options on the callable, for example
    with `functools.partial`. The suffix does not select the serializer.
    """
    suffix = "".join(Path(name).suffixes)
    stem = name.removesuffix(suffix)
    identifier = f"{prefix + ':' * bool(prefix)}{stem}{':' * bool(postfix) + postfix}"
    path = Path(writer.log_dir if isinstance(writer, SummaryWriter) else writer)
    path /= f"{identifier + '-' * bool(identifier)}{step}{suffix}"
    serializer(table, path)


def log_plot(
    step: int,
    writer: SummaryWriter,
    /,
    fig: Figure,
    *,
    rasterization_options: Mapping[str, Any] = EMPTY_MAP,
    name: str = "forecastplot",
    prefix: str = "",
    postfix: str = "",
) -> None:
    r"""Rasterize an existing Matplotlib figure and log it to TensorBoard.

    Args:
        step: The current step.
        fig: The Matplotlib figure to log.
        rasterization_options: Options to pass to `tsdm.plotting.rasterize`.
        writer: The writer to log to.
        name: The name of the plot.
        prefix: The prefix of the plot.
        postfix: The postfix of the plot.
    """
    identifier = f"{prefix + ':' * bool(prefix)}{name}{':' * bool(postfix) + postfix}"
    # rasterize the figure
    image = rasterize(fig, **rasterization_options)
    writer.add_image(f"{identifier}", image, step, dataformats="HWC")
