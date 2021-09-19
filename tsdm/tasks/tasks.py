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

import logging
from abc import ABC, abstractmethod
from typing import Final

from torch.utils.data import DataLoader

# from tsdm.losses import LOSSES


LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "BaseTask",
]


class BaseTask(ABC):
    r"""Abstract Base Class for Tasks.

    A task is always bound to a dataset.
    Its main task is create a dataloader that cycles through the test dataset once.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        r"""Perform preprocessing of the dataset."""

    @property
    @abstractmethod
    def dataset(self):
        r"""Return the preprocessed dataset."""

    @property
    @abstractmethod
    def train_dataset(self):
        r"""Return the preprocessed non-test data."""

    @property
    @abstractmethod
    def test_dataset(self):
        r"""Return the preprocessed test-data."""

    @property
    def train_loader(self):
        r"""Return a default trainloader (optional)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def test_loader(self) -> DataLoader:
        r"""Return a Dataloader instance that iterates throught the test set.

        For instance-wise metrics

        .. code-block:: python

            r = mean([test_metric(y, model(x)) for x,y in test_loader])

        For decomposable metrics:

        For non-decomposable metrics:

        .. code-block:: python

            pairs = [(y, model(x)) for x,y in test_loader(batch_size=1)]
            score = test_metric(pairs)
        """

    @property
    @abstractmethod
    def test_metric(self):
        r"""Return the test metric."""

    @property
    @abstractmethod
    def accumulation_function(self):
        r"""Return the accumulation function."""
