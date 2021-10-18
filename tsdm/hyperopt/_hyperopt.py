r"""Utilities for Hyperparameter-Optimization."""

from __future__ import annotations

__all__ = [
    # Constants
    "LookupTables",
    "ACTIVATIONS",
    "DATASETS",
    "ENCODERS",
    "LOSSES",
    "LR_SCHEDULERS",
    "METRICS",
    "MODELS",
    "OPTIMIZERS",
    "TASKS",
]

import logging
from typing import Final

from tsdm.datasets import DATASETS
from tsdm.encoders import ENCODERS
from tsdm.losses import LOSSES
from tsdm.metrics import METRICS
from tsdm.models import MODELS
from tsdm.optimizers import LR_SCHEDULERS, OPTIMIZERS
from tsdm.tasks import TASKS
from tsdm.util.activations import ACTIVATIONS
from tsdm.util.types import LookupTable

LOGGER = logging.getLogger(__name__)

LookupTables: Final[LookupTable[LookupTable]] = {
    "activation": ACTIVATIONS,
    "dataset": DATASETS,
    "preprocessor": ENCODERS,
    "loss": LOSSES,
    "lr_scheduler": LR_SCHEDULERS,
    "metric": METRICS,
    "model": MODELS,
    "optimizer": OPTIMIZERS,
    "task": TASKS,
}
r"""Dictionary of all available configurables."""
