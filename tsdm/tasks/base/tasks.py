r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.

Decomposable METRICS
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

Non-decomposable METRICS
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

Normal Encoder
--------------

A normal pre_encoder is an pre_encoder with the property that all output tensors
share the same index axis.

I.e. it has a signature of the form ``list[tensor[n, ...]] → list[tensor[n, ...]]``
Pre-Encoder: Map DataFrame to torch.util.data.Dataset



Default DataLoader Creation
---------------------------

.. code-block:: python

    data = pre_processor.encode(data)  # DataFrame to DataFrame
    data = pre_encoder.encode(data)  # DataFrame to DataSet
    dataset = TensorDataset(*inputs, targets)
    sampler = SequenceSampler(tuple[TimeTensor], tuple[StaticTensor])
    dataloader = DataLoader(dataset, sampler=sampler, collate=....)
    batch = next(dataloader)

    inputs,
"""

__all__ = [
    # Classes
    "BaseTask",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import Any, Generic, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tsdm.datasets import DATASET
from tsdm.encoders import ModularEncoder
from tsdm.util import LazyDict
from tsdm.util.types import KeyType

__logger__ = logging.getLogger(__name__)


class BaseTask(ABC, Generic[KeyType]):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    ``(inputs, targets)``. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes
    ----------
    index: list[str]
        A list of string specifying the data splits of interest.
    train_batch_size: int, default=32
        Default batch-size used by batchloader.
    eval_batch_size: int, default=128
        Default batch-size used by dataloaders (for evaluation).
    preprocessor: Optional[Encoder], default=None
        Task specific preprocessing. For example, the EVP might specifically ask for
        evaluation of Mean Squared Error on standardized data.
    dataset: Dataset
        The attached dataset
    splits: Mapping[KeyType, Any]
        Contains slices of the dataset. Contains a slice for each key, but may
        also hold additional entries. (For example: "joint" = "train"+"valid")
    batchloaders: Mapping[KeyType, DataLoader]
        The main DataLoader to be used for training models.
    dataloaders: Mapping[KeyType, DataLoader]
        Holds ``DataLoaders`` for all the index.
    """

    # __slots__ = ()  # https://stackoverflow.com/a/62628857/9318372

    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""
    preprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after batching)."""

    @property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""

    @property
    @abstractmethod
    def dataset(self) -> DATASET:
        r"""Return the cached dataset associated with the task."""

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyType]:
        r"""List of index."""

    @property
    @abstractmethod
    def splits(self) -> Mapping[KeyType, Any]:
        r"""Cache dictionary of dataset slices."""

    @cached_property
    def dataloaders(self) -> Mapping[KeyType, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        return LazyDict(
            (
                key,
                (
                    self.get_dataloader,
                    {
                        "key": key,
                        "batch_size": self.eval_batch_size,
                        "shuffle": False,
                        "drop_last": False,
                    },
                ),
            )
            for key in self.splits
        )

    @cached_property
    def batchloaders(self) -> Mapping[KeyType, DataLoader]:
        r"""Cache main training-dataloader."""
        return LazyDict(
            (
                key,
                (
                    self.get_dataloader,
                    {
                        "key": key,
                        "batch_size": self.train_batch_size,
                        "shuffle": True,
                        "drop_last": True,
                    },
                ),
            )
            for key in self.splits
        )

    @abstractmethod
    def get_dataloader(
        self,
        key: KeyType,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split.

        Parameters
        ----------
        key: str
            From which part of the dataset to construct the loader
        batch_size: int = 32
        shuffle: bool = True
            Whether to get samples in random order.
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
            defaults to cuda if cuda is available.
        kwargs:
            Options to be passed directly to the dataloader such as the generator.

        Returns
        -------
        DataLoader
        """

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        string = (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset.name}, "
            f"test_metric={type(self.test_metric).__name__})"
        )
        return string
