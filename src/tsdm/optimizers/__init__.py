r"""Utilities for optimizers."""

__all__ = [
    # Constants
    "LR_SCHEDULERS",
    "OPTIMIZERS",
    # ABCs & Protocols
    "LRScheduler",
    "Optimizer",
    "TorchLRScheduler",
    "TorchOptimizer",
]

from torch import optim
from torch.optim import Optimizer as TorchOptimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler as TorchLRScheduler

from tsdm.optimizers.base import LRScheduler, Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    "ASGD"       : optim.ASGD,
    "Adadelta"   : optim.Adadelta,
    "Adagrad"    : optim.Adagrad,
    "Adam"       : optim.Adam,
    "AdamW"      : optim.AdamW,
    "Adamax"     : optim.Adamax,
    "LBFGS"      : optim.LBFGS,  # type: ignore[dict-item]
    "RMSprop"    : optim.RMSprop,
    "Rprop"      : optim.Rprop,
    "SGD"        : optim.SGD,
    "SparseAdam" : optim.SparseAdam,
}  # fmt: skip
r"""Dictionary of all available optimizers."""

LR_SCHEDULERS: dict[str, type[LRScheduler]] = {
    "CosineAnnealingLR"           : lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts" : lr_scheduler.CosineAnnealingWarmRestarts,
    "CyclicLR"                    : lr_scheduler.CyclicLR,
    "ExponentialLR"               : lr_scheduler.ExponentialLR,
    "LambdaLR"                    : lr_scheduler.LambdaLR,
    "MultiStepLR"                 : lr_scheduler.MultiStepLR,
    "MultiplicativeLR"            : lr_scheduler.MultiplicativeLR,
    "OneCycleLR"                  : lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau"           : lr_scheduler.ReduceLROnPlateau,  # type: ignore[dict-item]
    "StepLR"                      : lr_scheduler.StepLR,
}  # fmt: skip
r"""Dictionary of all available lr_schedulers."""

del lr_scheduler, optim
