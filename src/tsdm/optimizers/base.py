r"""Protocol for optimizers."""

__all__ = [
    # Protocols
    "Optimizer",
    "LRScheduler",
]

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol

import torch
from torch import Tensor

type StateDict = dict[str, Any]


class Optimizer(Protocol):
    r"""Protocol version of `torch.optim.Optimizer`."""

    defaults: dict[str, Any]
    param_groups: list[dict[str, Any]]
    state: defaultdict[Tensor, dict[str, Tensor]]

    def add_param_group(self, param_group: dict[str, Any]) -> None: ...
    def state_dict(self) -> StateDict: ...
    def load_state_dict(self, state_dict: StateDict) -> None: ...
    def step(self, closure: Any = None) -> Tensor | None: ...
    def zero_grad(self, *, set_to_none: bool = True) -> None: ...


class LRScheduler(Protocol):
    r"""Protocol version of `torch.optim.lr_scheduler.LRScheduler`."""

    base_lrs: list[Tensor | float]
    last_epoch: int = -1
    optimizer: Final[Optimizer]  # type: ignore[misc]

    def state_dict(self) -> StateDict: ...
    def load_state_dict(self, state_dict: StateDict) -> None: ...
    def step(self, epoch: Optional[int] = None) -> None: ...
    def get_last_lr(self) -> list[Tensor | float]: ...
    def get_lr(self) -> list[Tensor | float]: ...


if TYPE_CHECKING:
    # ensure that the Protocols are compatible with the actual classes
    TorchOptimizer: type[Optimizer] = torch.optim.Optimizer
    TorchLRScheduler: type[LRScheduler] = torch.optim.lr_scheduler.LRScheduler
