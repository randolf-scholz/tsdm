r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a split to cater to the question of
forecasting horizons.

Decomposable Metrics
--------------------

**Example:** Mean Square Error (MSE)

.. code-block:: python

    dims = (1, ...)  # sum over all axes except batch dimension
    # y, yhat are of shape (B, ...)
    test_metric = lambda y, yhat: torch.sum( (y-yhat)**2, dim=dims )
    accumulation = torch.mean

**Recipe**

.. code-block:: python

    r = []
    for x, y in dataloader:
        r.append( test_metric(y, model(x)) )

    score = accumulation(torch.concat(r, dim=BATCHDIM))

Non-decomposable Metrics
------------------------

**Example:** Area Under the Receiver Operating Characteristic Curve (AUROC)

test_metric = torch.AUROC()   # expects two tensors of shape (N, ...) or (N, C, ...)

.. code-block:: python

   score = test_metric([(y, model(x)) for x, y in test_loader])
   accumulation = None or identity function (tbd.)

**Recipe**

.. code-block:: python

    ys, yhats = []
    for x, y in dataloader:
        ys.append( y )
        yhats.append( model(x) )

    ys = torch.concat(ys, dim=BATCHDIM)
    yhats = torch.concat(yhats, dim=BATCHDIM)
    score = test_metric(ys, yhats)
"""

from __future__ import annotations

__all__ = [
    # Classes
    "BaseTask",
]

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tsdm.datasets import Dataset
from tsdm.losses import Loss

LOGGER = logging.getLogger(__name__)


class BaseTask(ABC):
    r"""Abstract Base Class for Tasks.

    A task is always bound to a dataset.
    Its main task is create a dataloader that cycles through the test dataset once.
    """

    # __slots__ = ()  # https://stackoverflow.com/a/62628857/9318372
    dataset: Dataset
    """The whole dataset."""
    train_dataset: Dataset
    """The test-slice of the datset."""
    trial_dataset: Dataset
    """The test-slice of the datset."""
    test_metric: Loss
    """The test-metric (usually instance-wise)."""
    accumulation_function: Callable[..., Tensor]
    """The accumulation function (usually sum or mean or identity)."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        r"""Perform preprocessing of the dataset."""
        super().__init__()

    @abstractmethod
    def get_dataloader(
        self,
        split,  # noqa   # ignore the missing type hint ...
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DataLoader:
        r"""Return a dataloder object for the specified split.

        Parameters
        ----------
        split: str
            From which part of the dataset to construct the loader
        batch_size: int = 32
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
            defaults to cuda if cuda is available.

        Returns
        -------
        DataLoader
        """
