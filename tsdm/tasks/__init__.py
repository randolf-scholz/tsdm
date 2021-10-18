r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a split to cater to the question of
forecasting horizons.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "Task",
    "TASKS",
    # Classes
    "BaseTask",
    # Tasks
    "ETDatasetInformer",
]


import logging
from typing import Final

from tsdm.tasks.etdataset import ETDatasetInformer
from tsdm.tasks.tasks import BaseTask
from tsdm.util.types import LookupTable

LOGGER = logging.getLogger(__name__)


Task = BaseTask
r"""Type hint for tasks."""

TASKS: Final[LookupTable[type[Task]]] = {
    "ETDatasetInformer": ETDatasetInformer,
}
r"""Dictionary of all available tasks."""
