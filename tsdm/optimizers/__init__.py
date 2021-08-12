r"""Utilities for optimizers."""


import logging
from typing import Final, Type

import torch.optim
from torch.optim import lr_scheduler, Optimizer

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["OPTIMIZERS", "LR_SCHEDULERS"]


OPTIMIZERS: Final[dict[str, Type[Optimizer]]] = {
    "Adadelta": torch.optim.Adadelta,
    # Implements Adadelta algorithm.
    "Adagrad": torch.optim.Adagrad,
    # Implements Adagrad algorithm.
    "Adam": torch.optim.Adam,
    # Implements Adam algorithm.
    "AdamW": torch.optim.AdamW,
    # Implements AdamW algorithm.
    "SparseAdam": torch.optim.SparseAdam,
    # Implements lazy version of Adam algorithm suitable for sparse tensors.
    "Adamax": torch.optim.Adamax,
    # Implements Adamax algorithm (a variant of Adam based on infinity norm).
    "ASGD": torch.optim.ASGD,
    # Implements Averaged Stochastic Gradient Descent.
    "LBFGS": torch.optim.LBFGS,
    # Implements L-BFGS algorithm, heavily inspired by minFunc.
    "RMSprop": torch.optim.RMSprop,
    # Implements RMSprop algorithm.
    "Rprop": torch.optim.Rprop,
    # Implements the resilient backpropagation algorithm.
    "SGD": torch.optim.SGD,
    # Implements stochastic gradient descent (optionally with momentum).
}
r"""Utility dictionary, for use in model creation from Hyperparameter dicts."""

LR_SCHEDULER = lr_scheduler._LRScheduler
LR_SCHEDULERS: Final[dict[str, Type[LR_SCHEDULER]]] = {
    "LambdaLR": lr_scheduler.LambdaLR,
    "MultiplicativeLR": lr_scheduler.MultiplicativeLR,  # type: ignore
    "StepLR": lr_scheduler.StepLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    # "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,   # not subclass of _LRScheduler...
    "CyclicLR": lr_scheduler.CyclicLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,  # type: ignore
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
}
r"""Utility dictionary, for use in model creation from Hyperparameter dicts."""
