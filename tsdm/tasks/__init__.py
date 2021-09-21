r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a split to cater to the question of
forecasting horizons.
"""


from __future__ import annotations

import logging
from typing import Final, Type

from tsdm.tasks.ETDataset import ETDatasetInformer
from tsdm.tasks.tasks import BaseTask

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = ["Task", "TASKS"] + [
    "ETDatasetInformer",
]


Task = Type[BaseTask]
r"""Type hint for tasks."""

TASKS: Final[dict[str, Task]] = {
    "ETDatasetInformer": ETDatasetInformer,
}
r"""Dictionary containing all available tasks."""
