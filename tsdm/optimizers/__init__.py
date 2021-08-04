r"""Utilities for optimizers."""


import logging
from typing import Final, Type

import torch
from torch.optim import Optimizer


logger = logging.getLogger(__name__)
__all__ = ["OPTIMIZERS"]

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
