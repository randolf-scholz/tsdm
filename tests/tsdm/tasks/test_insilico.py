r"""Test task implementation with InSilico."""

from pandas import DataFrame, MultiIndex
from torch.utils.data import DataLoader

from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import InSilicoTask
from tsdm.timeseries import PandasTSC, TimeSeriesSampleGenerator


def test_insilico_task() -> None:
    r"""Test the TimeSeriesDatasetTask."""
    split_id = (0, "train")
    task = InSilicoTask()
    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.index, MultiIndex)
    assert isinstance(task.splits[split_id], PandasTSC)
    assert isinstance(task.samplers[split_id], HierarchicalSampler)
    assert isinstance(task.generators[split_id], TimeSeriesSampleGenerator)
    assert isinstance(task.dataloaders[split_id], DataLoader)
    assert task.collate_fns[split_id] is NotImplemented
    assert isinstance(task.train_split, dict)

    sampler = task.samplers[split_id]
    key = next(iter(sampler))
    assert isinstance(key, tuple)
    generator = task.generators[split_id]
    print("key", type(key), key)
    print("generator", type(generator), generator)
    sample = generator[key]
    assert isinstance(sample, tuple)

    dataloader = task.dataloaders[split_id]
    batch = next(iter(dataloader))
    assert batch is not None
