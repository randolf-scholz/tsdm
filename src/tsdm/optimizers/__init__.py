r"""Utilities for optimizers."""

__all__ = [
    # Constants
    "Optimizer",
    "OPTIMIZERS",
    # Classes
    "LRScheduler",
    "LR_SCHEDULERS",
]

from typing import Final, TypeAlias

from torch import optim
from torch.optim import lr_scheduler

Optimizer: TypeAlias = optim.Optimizer
r"""Type hint for optimizers."""
LRScheduler = lr_scheduler.LRScheduler
r"""Type hint for lr_schedulers."""


OPTIMIZERS: Final[dict[str, type[Optimizer]]] = {
    "Adadelta": optim.Adadelta,
    # Implements Adadelta algorithm.
    "Adagrad": optim.Adagrad,
    # Implements Adagrad algorithm.
    "Adam": optim.Adam,
    # Implements Adam algorithm.
    "AdamW": optim.AdamW,
    # Implements AdamW algorithm.
    "SparseAdam": optim.SparseAdam,
    # Implements lazy version of Adam algorithm suitable for sparse tensors.
    "Adamax": optim.Adamax,
    # Implements Adamax algorithm (a variant of Adam based on infinity norm).
    "ASGD": optim.ASGD,
    # Implements Averaged Stochastic Gradient Descent.
    "LBFGS": optim.LBFGS,
    # Implements L-BFGS algorithm, heavily inspired by minFunc.
    "RMSprop": optim.RMSprop,
    # Implements RMSprop algorithm.
    "Rprop": optim.Rprop,
    # Implements the resilient backpropagation algorithm.
    "SGD": optim.SGD,
    # Implements stochastic gradient descent (optionally with momentum).
}
r"""Dictionary of all available optimizers."""


LR_SCHEDULERS: Final[dict[str, type[LRScheduler]]] = {
    "LambdaLR": lr_scheduler.LambdaLR,
    "MultiplicativeLR": lr_scheduler.MultiplicativeLR,
    "StepLR": lr_scheduler.StepLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    # "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,   # not subclass of _LRScheduler...
    "CyclicLR": lr_scheduler.CyclicLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
}
r"""Dictionary of all available lr_schedulers."""

del Final, TypeAlias, lr_scheduler, optim
