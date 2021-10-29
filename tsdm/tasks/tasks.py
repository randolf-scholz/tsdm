r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a split to cater to the question of
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

A normal pre_encoder is an pre_encoder with the property that all output tensors share the same index axis.

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

from __future__ import annotations

__all__ = [
    # Classes
    "BaseTask",
]

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tsdm.datasets import Dataset
from tsdm.encoders import Encoder

LOGGER = logging.getLogger(__name__)


class BaseTask(ABC):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    Attributes
    ----------
    keys: list[str]
        A list of string specifying the data splits of interest.
    train_batch_size: int, default=32
        Default batch-size used by batchloader.
    eval_batch_size: int, default=128
        Default batch-size used by dataloaders (for evaluation).
    preprocessor: Optional[Encoder], default=None
        Task specific preprocessing. For example, the EVP might specifically ask for
        evaluation of Mean Squared Error on standardized data.
    pre_encoder: Optional[Encoder], default=None
        Model specific pre_encoder. Must be a :ref:`normal pre_encoder <Normal Encoder>`.
        Is applied before batching and caches results.
    dataset: Dataset
        The attached dataset
    splits: dict[keys, Dataset]
        Contains slices of the dataset. Contains a slice for each key, but may
        also hold additional entries. (For example: "joint" = "train"+"valid")
    batchloader: DataLoader
        The main DataLoader to be used for training models.
    dataloaders: dict[keys, DataLoader]
        Holds ``DataLoaders`` for all the keys.
    """

    # __slots__ = ()  # https://stackoverflow.com/a/62628857/9318372
    KEYS: Any
    r"""Should be Literal[tuple(str)]"""
    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""
    preprocessor: Optional[Encoder] = None
    r"""Optional task specific preprocessor."""
    pre_encoder: Optional[Encoder] = None
    r"""Optional model specific normal pre_encoder that is applied before batching."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        r"""Perform preprocessing of the dataset."""
        super().__init__()

    # @property
    # @abstractmethod
    # def accumulation_function(self) -> Callable[[Tensor], Tensor]:
    #     r"""Accumulates residuals into loss - usually mean or sum."""

    @cached_property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The target metric."""

    @cached_property
    @abstractmethod
    def keys(self) -> list[str]:
        r"""List of keys."""

    @cached_property
    @abstractmethod
    def dataset(self) -> Dataset:
        r"""Cache the dataset."""

    @cached_property
    @abstractmethod
    def splits(self) -> dict[KEYS, Any]:
        r"""Cache dictionary of dataset slices."""

    @cached_property
    def dataloaders(self) -> dict[KEYS, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        return {
            key: self.get_dataloader(
                key, batch_size=self.eval_batch_size, shuffle=False, drop_last=False
            )
            for key in self.splits
        }

    @cached_property
    def batchloader(self) -> DataLoader:
        r"""Cache main training-dataloader."""
        return self.get_dataloader(
            "train", batch_size=self.train_batch_size, shuffle=True, drop_last=True
        )

    @abstractmethod
    def get_dataloader(
        self,
        split: KEYS,
        batch_size: int = 1,
        shuffle: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split.

        Parameters
        ----------
        split: str
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
