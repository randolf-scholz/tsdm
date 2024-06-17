r"""Utilities for logging."""

__all__ = [
    "AdamState",
    "TargetsAndPredics",
    # Functions
    "compute_metrics",
    "eval_metric",
    "save_checkpoint",
    "yield_optimizer_params",
]

import pickle
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
import yaml
from torch import Tensor, jit, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import Any, NamedTuple, TypedDict

from tsdm.metrics import LOSSES, Metric
from tsdm.types.aliases import FilePath


class AdamState(TypedDict):
    r"""Adam optimizer state."""

    step: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor


def yield_optimizer_params(optimizer: Optimizer, /) -> Iterator[nn.Parameter]:
    r"""Get parameters from optimizer."""
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.requires_grad:
                yield param


class TargetsAndPredics(NamedTuple):
    r"""Targets and predictions."""

    targets: Tensor
    predics: Tensor


@torch.no_grad()
def eval_metric(
    metric: str | Metric | type[Metric],
    /,
    *,
    targets: Tensor,
    predics: Tensor,
) -> Tensor:
    r"""Evaluate a metric."""
    match metric:
        case str(metric_name):
            _metric = LOSSES[metric_name]
            return eval_metric(_metric, targets=targets, predics=predics)
        case type() as metric_type:
            metric_func = metric_type()
            return eval_metric(metric_func, targets=targets, predics=predics)
        case Metric() as func:
            return func(targets, predics)
        case _:
            raise TypeError(f"{type(metric)=} not understood!")


@torch.no_grad()
def compute_metrics(
    metrics: (
        str
        | Metric
        | type[Metric]
        | Sequence[str | Metric | type[Metric]]
        | Mapping[str, str | Metric | type[Metric]]
    ),
    /,
    *,
    targets: Tensor,
    predics: Tensor,
) -> dict[str, Tensor]:
    r"""Compute multiple metrics."""
    match metrics:
        case str(name):
            return {name: eval_metric(LOSSES[name], targets=targets, predics=predics)}
        case type() as cls:
            return {cls.__name__: eval_metric(cls, targets=targets, predics=predics)}
        case Metric() as func:
            return {func.__class__.__name__: func(targets, predics)}
        case Sequence() as sequence:
            results: dict[str, Tensor] = {}
            for metric in sequence:
                results |= compute_metrics(metric, targets=targets, predics=predics)
        case Mapping() as mapping:
            return {
                key: eval_metric(metric, targets=targets, predics=predics)
                for key, metric in mapping.items()
            }
        case _:
            raise TypeError(f"{type(metrics)=} not understood!")
    return results


def save_checkpoint(step: int, path: FilePath, *, objects: Mapping[str, Any]) -> None:
    r"""Save checkpoints of given paths."""
    path = Path(path) / f"{step}"
    path.mkdir(parents=True, exist_ok=True)

    for name, obj in dict(objects).items():
        match obj:
            case None:
                pass
            case jit.ScriptModule():
                jit.save(obj, path / name)
            case nn.Module():
                torch.save(obj, path / name)
            case Optimizer():
                torch.save(obj, path / name)
            case LRScheduler():
                torch.save(obj, path / name)
            case dict() | list() | tuple() | set() | str() | int() | float() | None:
                with open(path / f"{name}.yaml", "w", encoding="utf8") as file:
                    yaml.safe_dump(obj, file)
            case _:
                with open(path / f"{name}.pickle", "wb") as file:
                    pickle.dump(obj, file)
