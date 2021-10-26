r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a split to cater to the question of
forecasting horizons.

The whole start-to-end data pipeline looks as follows:

Modus Operandi: small data-set, everything in-memory

1. Raw Dataset (numpy/pandas/xarray/etc.)
2. Perform task specific pre-processing.
3. Perform the model specific encoding
4. Sample a batch
   - batch should consist of model_inputs, model_targets, true_targets
   - targets are model/loss specific preprocessed targets from raw dataset, for instance
     one-hot encoded for classification task
5. compute:
   - `outputs = model(*inputs)`
        - outputs should have the same shape as targets ?!?
        - but then the model needs to know task specific things!
        - but most models need to know these during initialization anyway.
        - but our model here doesn't and that makes it special.
        - alternatively another encoding/decoding layer is necessary.
   - `predictions = ???(outputs)`
        - convert to "true" targets
        - i.e. revert any encoding/preprocessing steps

6


Modus Operandi: large data-set, stream from disk





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
