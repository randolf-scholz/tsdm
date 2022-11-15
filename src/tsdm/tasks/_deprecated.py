"""Deprecated Task code."""

from __future__ import annotations

__all__ = ["BaseTask"]

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler as TorchSampler

from tsdm.datasets import Dataset
from tsdm.encoders import ModularEncoder
from tsdm.tasks.base import BaseTaskMetaClass
from tsdm.utils import LazyDict
from tsdm.utils.types import KeyVar


class BaseTask(ABC, Generic[KeyVar], metaclass=BaseTaskMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes
    ----------
    train_batch_size: int, default 32
        Default batch-size used by batchloader.
    eval_batch_size: int, default 128
        Default batch-size used by dataloaders (for evaluation).
    dataset: Dataset
        The attached dataset
    """

    # ABCs should have slots https://stackoverflow.com/a/62628857/9318372

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    preprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before sampling/batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after sampling/batching)."""

    dataloader_config_train = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    dataloader_config_infer = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    def __init__(
        self,
        dataset: TorchDataset,
        sampler: TorchSampler,
        dataloader_config_train: dict[str, Any],
        **dataloader_kwargs: Any,
    ):
        r"""Initialize the task object."""
        self.dataloader_config_train |= dataloader_kwargs
        self.dataloader_config_infer |= dataloader_kwargs

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        string = (
            f"{self.__class__.__name__}("
            # f"dataset={self.dataset.name}, "
            f"test_metric={type(self.test_metric).__name__})"
        )
        return string

    @property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DataFrame:
        r"""Return the cached dataset associated with the task."""

    def split_type(self, key: KeyVar) -> Literal["train", "infer", "unknown"]:
        r"""Return the type of split."""
        train_patterns = ["train", "training"]
        infer_patterns = [
            "test",
            "testing",
            "val",
            "valid",
            "validation",
            "eval",
            "evaluation",
            "infer",
            "inference",
        ]

        if isinstance(key, str):
            if key.lower() in train_patterns:
                return "train"
            if key.lower() in infer_patterns:
                return "infer"
        if isinstance(key, Sequence):
            patterns = {self.split_type(k) for k in key}
            if patterns <= {"train", "unknown"}:
                return "train"
            if patterns <= {"infer", "unknown"}:
                return "infer"
            if patterns == {"unknown"}:
                return "unknown"
            raise ValueError(f"{key=} contains both train and infer splits.")
        return "unknown"

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyVar]:
        r"""List of index."""

    @abstractmethod
    def make_split(self, key: KeyVar, /) -> TorchDataset:
        r"""Return the split associated with the given key."""

    @abstractmethod
    def make_sampler(self, key: KeyVar, /) -> TorchSampler:
        r"""Create sampler for the given split."""

    def make_dataloader(
        self,
        key: KeyVar,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split."""
        dataset = self.splits[key]
        sampler = self.samplers[key]

        match self.split_type(key):
            case "train":
                kwargs = self.dataloader_configs["train"]
            case "infer":
                kwargs = self.dataloader_configs["train"]
            case _:
                raise ValueError(f"Unknown split type: {self.split_type(key)=}")

        kwargs = kwargs | dataloader_kwargs
        dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        return dataloader

    @cached_property
    def dataloader_configs(self) -> dict[str, dict[str, Any]]:
        r"""Return dataloader configuration."""
        return {
            "train": self.dataloader_config_train,
            "eval": self.dataloader_config_infer,
        }

    @cached_property
    def splits(self) -> Mapping[KeyVar, TorchDataset]:
        r"""Cache dictionary of dataset splits."""
        return LazyDict({k: self.make_split for k in self.splits})

    @cached_property
    def samplers(self) -> Mapping[KeyVar, TorchSampler]:
        r"""Return a dictionary of samplers for each split."""
        return LazyDict({k: self.make_sampler for k in self.splits})

    @cached_property
    def dataloaders(self) -> Mapping[KeyVar, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        return LazyDict({k: self.make_dataloader for k in self.splits})
