r"""Test task implementation with InSilico."""

from pandas import DataFrame, MultiIndex
from torch.utils.data import DataLoader

from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import InSilicoTask
from tsdm.timeseries import PandasTSC, TimeSeriesSampleGenerator


def test_insilico_task(SplitID: tuple[int, str] = (0, "train")) -> None:
    r"""Test the TimeSeriesDatasetTask."""
    task = InSilicoTask()
    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.index, MultiIndex)
    assert isinstance(task.splits[SplitID], PandasTSC)
    assert isinstance(task.samplers[SplitID], HierarchicalSampler)
    assert isinstance(task.generators[SplitID], TimeSeriesSampleGenerator)
    assert isinstance(task.dataloaders[SplitID], DataLoader)
    assert task.collate_fns[SplitID] is NotImplemented
    assert isinstance(task.train_split, dict)

    sampler = task.samplers[SplitID]
    key = next(iter(sampler))
    assert isinstance(key, tuple)
    generator = task.generators[SplitID]
    print("key", type(key), key)
    print("generator", type(generator), generator)
    sample = generator[key]
    assert isinstance(sample, tuple)

    dataloader = task.dataloaders[SplitID]
    batch = next(iter(dataloader))
    assert batch is not None
