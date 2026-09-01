r"""Test task implementation with InSilico."""

from pandas import DataFrame
from torch.utils.data import DataLoader

from tsdm.datatools import CallableDataset
from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import InSilicoTask
from tsdm.timeseries.pandas import SplitTimeData, TimeSeriesCollection


def test_insilico_task() -> None:
    r"""Test the TimeSeriesDatasetTask."""
    split_id = (0, "train")
    task = InSilicoTask()
    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.splits[split_id], TimeSeriesCollection)
    assert isinstance(task.samplers[split_id], HierarchicalSampler)
    assert isinstance(task.generators[split_id], CallableDataset)
    assert isinstance(task.dataloaders[split_id], DataLoader)
    assert task.collate_fns[split_id] is NotImplemented
    assert task.get_train_split(split_id) == split_id

    sampler = task.samplers[split_id]
    key = next(iter(sampler))
    assert isinstance(key, tuple)
    generator = task.generators[split_id]
    print("key", type(key), key)
    print("generator", type(generator), generator)
    sample = generator[key]
    assert isinstance(sample, SplitTimeData)

    dataloader = task.dataloaders[split_id]
    batch = next(iter(dataloader))
    assert batch is not None
