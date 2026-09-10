r"""Utilities for logging."""

__all__ = [
    "TargetsAndPredictions",
    # Functions
    "compute_metrics",
    "eval_metric",
    "save_checkpoint",
    "yield_optimizer_params",
]

import pickle
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tsdm.metrics import LOSSES, Loss
from tsdm.types.aliases import FilePath


def yield_optimizer_params(optimizer: Optimizer, /) -> Iterator[nn.Parameter]:
    r"""Yield the parameters registered to an optimizer."""
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.requires_grad:
                yield param


class TargetsAndPredictions(NamedTuple):
    r"""Targets and predictions."""

    targets: Tensor
    predictions: Tensor


@torch.no_grad()
def eval_metric(
    metric: str | Loss | type[Loss],
    /,
    *,
    predictions: Tensor,
    targets: Tensor,
) -> Tensor:
    r"""Evaluate a metric."""
    match metric:
        case str(metric_name):
            metric_ = LOSSES[metric_name]
            return eval_metric(metric_, predictions=predictions, targets=targets)
        case type() as metric_type:
            metric_func = metric_type()
            return eval_metric(metric_func, predictions=predictions, targets=targets)
        case Callable() as func:
            return func(predictions=predictions, targets=targets)
        case _:
            raise TypeError(f"{type(metric)=} not understood!")


@torch.no_grad()
def compute_metrics(
    metrics: (
        str
        | Loss
        | type[Loss]
        | Sequence[str | Loss | type[Loss]]
        | Mapping[str, str | Loss | type[Loss]]
    ),
    /,
    *,
    predictions: Tensor,
    targets: Tensor,
) -> dict[str, Tensor]:
    r"""Compute multiple metrics."""
    match metrics:
        case str(name):
            return {
                name: eval_metric(
                    LOSSES[name], predictions=predictions, targets=targets
                )
            }
        case type() as cls:
            return {
                cls.__name__: eval_metric(cls, predictions=predictions, targets=targets)
            }
        case Callable() as func:
            return {
                func.__class__.__name__: func(predictions=predictions, targets=targets)
            }
        case Sequence() as sequence:
            results: dict[str, Tensor] = {}
            for metric in sequence:
                results |= compute_metrics(
                    metric, predictions=predictions, targets=targets
                )
        case Mapping() as mapping:
            return {
                key: eval_metric(metric, predictions=predictions, targets=targets)
                for key, metric in mapping.items()
            }
        case _:
            raise TypeError(f"{type(metrics)=} not understood!")
    return results


def save_checkpoint(step: int, path: FilePath, *, objects: Mapping[str, Any]) -> None:
    r"""Save checkpoints of given paths."""
    path = Path(path) / f"{step}"
    path.mkdir(parents=True, exist_ok=True)

    for name, obj in objects.items():
        match obj:
            case None:
                pass
            case torch.jit.ScriptModule():
                torch.jit.save(obj, path / name)
            case nn.Module():
                torch.save(obj, path / name)
            case Optimizer():
                torch.save(obj, path / name)
            case LRScheduler():
                torch.save(obj, path / name)
            case dict() | list() | tuple() | set() | str() | int() | float() | None:
                path /= f"{name}.yaml"
                with path.open("w", encoding="utf8") as file:
                    yaml.safe_dump(obj, file)
            case _:
                path /= f"{name}.pickle"
                with path.open("wb") as file:
                    pickle.dump(obj, file)
