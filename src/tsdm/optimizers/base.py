r"""Protocol for optimizers."""

__all__ = [
    "Optimizer",
    "LRScheduler",
]

from collections import defaultdict
from typing import Any, Final, Optional, Protocol

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
    # FIXME: torch==2.4 has some bad type hints here...
    def step(self, closure: Any = None) -> Tensor | None: ...
    def zero_grad(self, *, set_to_none: bool = True) -> None: ...


class LRScheduler(Protocol):
    r"""Protocol version of `torch.optim.lr_scheduler.LRScheduler`."""

    base_lrs: list[float]
    last_epoch: int = -1
    optimizer: Final[Optimizer]  # type: ignore[misc]

    def state_dict(self) -> StateDict: ...
    def load_state_dict(self, state_dict: StateDict) -> None: ...
    def step(self, epoch: Optional[int] = None) -> None: ...
    def get_last_lr(self) -> list[float]: ...
    def get_lr(self) -> list[float]: ...
