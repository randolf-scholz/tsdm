# pyright: ignore
# mypy: ignore-errors
"""Deprecated Task code."""


__all__ = ["BaseTask", "OldBaseTask"]

import logging
import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property

from pandas import DataFrame
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset as TorchDataset,
    Sampler as TorchSampler,
)
from typing_extensions import Any, ClassVar, Generic, Literal, Optional, Protocol

from tsdm.datasets import Dataset
from tsdm.encoders import Encoder
from tsdm.types.variables import key_var as K
from tsdm.utils import LazyDict


class BaseDatasetMetaClass(type(Protocol)):  # type: ignore[misc]
    r"""Metaclass for BaseDataset."""

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwds: Any,  # pyright: ignore
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")


class BaseTask(Generic[K], metaclass=BaseDatasetMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes:
        dataset (Dataset): The attached dataset
    """

    # ABCs should have slots https://stackoverflow.com/a/62628857/9318372

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    preprocessor: Optional[Encoder] = None
    r"""Optional task specific preprocessor (applied before sampling/batching)."""
    postprocessor: Optional[Encoder] = None
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

    def split_type(self, key: K | Sequence[K]) -> Literal["train", "infer", "unknown"]:
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
            raise ValueError(f"{key=} is neither train nor infer split.")
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
    def index(self) -> Sequence[K]:
        r"""List of index."""

    @abstractmethod
    def make_split(self, key: K, /) -> TorchDataset:
        r"""Return the split associated with the given key."""

    @abstractmethod
    def make_sampler(self, key: K, /) -> TorchSampler:
        r"""Creates sampler for the given split."""

    def make_dataloader(
        self,
        key: K,
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
    def splits(self) -> Mapping[K, TorchDataset]:
        r"""Cache dictionary of dataset splits."""
        return LazyDict({k: self.make_split for k in self.splits})

    @cached_property
    def samplers(self) -> Mapping[K, TorchSampler]:
        r"""Return a dictionary of samplers for each split."""
        return LazyDict({k: self.make_sampler for k in self.splits})

    @cached_property
    def dataloaders(self) -> Mapping[K, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        return LazyDict({k: self.make_dataloader for k in self.splits})


class OldBaseTask(Generic[K], metaclass=BaseDatasetMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes:
        train_batch_size (int): Batch-size used by batchloader.
        eval_batch_size (int): Default batch-size used by dataloaders (for evaluation).
        dataset (Dataset): The attached dataset
    """

    # __slots__ = ()  # https://stackoverflow.com/a/62628857/9318372

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""
    preprocessor: Optional[Encoder] = None
    r"""Optional task specific preprocessor (applied before batching)."""
    postprocessor: Optional[Encoder] = None
    r"""Optional task specific postprocessor (applied after batching)."""

    def __init__(self) -> None:
        warnings.warn("deprecated, use new class", DeprecationWarning, stacklevel=1)

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
        ...

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DataFrame:
        r"""Return the cached dataset associated with the task."""
        ...

    @property
    @abstractmethod
    def index(self) -> Sequence[K]:
        r"""List of index."""
        ...

    @property
    @abstractmethod
    def splits(self) -> Mapping[K, Any]:
        r"""Cache dictionary of dataset slices."""
        ...

    @abstractmethod
    def make_dataloader(
        self,
        key: K,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split.

        Args:
            key: From which part of the dataset to construct the loader.
            dataloader_kwargs: Options to be passed directly to the dataloader such as the generator.

        Returns:
            DataLoader: A DataLoader for the selected key.
        """
        ...

    @cached_property
    def dataloaders(self) -> Mapping[Any, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        kwargs: dict[Any, Any] = {
            # "key": key,
            "batch_size": self.eval_batch_size,
            "shuffle": False,
            "drop_last": False,
        }

        return LazyDict({
            key: (self.make_dataloader, (key,), kwargs) for key in self.splits
        })
