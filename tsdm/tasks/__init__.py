r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a split to cater to the question of
forecasting horizons.


Idea:

The Pre-Encoder must work in the following way:

- ``(named)tuple[TimeTensor] → (named)tuple[TimeTensor]`` row-wise!
- ``(named)tuple[Tensor] → (named)tuple[Tensor]``

More generally, eligible inputs are:

- ``DataFrame``, ``TimeTensor``, ``tuple[DataFrame]``, ``tuple[TimeTensor]``
- Product-types.

Must return a `NamedTuple` that agrees with the original column names!
This allows us to select
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

__logger__ = logging.getLogger(__name__)


Task = BaseTask
r"""Type hint for tasks."""

TASKS: Final[LookupTable[type[Task]]] = {
    "ETDatasetInformer": ETDatasetInformer,
}
r"""Dictionary of all available tasks."""
