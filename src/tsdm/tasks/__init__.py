r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.


Idea:

The Pre-Encoder must work in the following way:

- `tuple[TimeTensor] → tuple[TimeTensor]` row-wise!
- `tuple[Tensor] → tuple[Tensor]`

More generally, eligible inputs are:

- `DataFrame`, `TimeTensor`, `tuple[DataFrame]`, `tuple[TimeTensor]`
- Product-types.

Must return a `NamedTuple` that agrees with the original column names!
This allows us to select
"""

__all__ = [
    # Sub- Packages
    "base",
    # Constants
    "Task",
    "TASKS",
    # Classes
    # Tasks
    "ETDatasetTask_Informer",
    "KIWI_RUNS_TASK",
    "KIWI_FINAL_PRODUCT",
]


import logging
from typing import Final

from tsdm.tasks import base
from tsdm.tasks.base import BaseTask
from tsdm.tasks.etdataset import ETDatasetTask_Informer
from tsdm.tasks.kiwi_final_product import KIWI_FINAL_PRODUCT
from tsdm.tasks.kiwi_runs_task import KIWI_RUNS_TASK

__logger__ = logging.getLogger(__name__)


Task = BaseTask
r"""Type hint for tasks."""

TASKS: Final[dict[str, type[Task]]] = {
    "ETDatasetTask_Informer": ETDatasetTask_Informer,
    "KIWI_RUNS_TASK": KIWI_RUNS_TASK,
    "KIWI_FINAL_PRODUCT": KIWI_FINAL_PRODUCT,
}
r"""Dictionary of all available tasks."""
