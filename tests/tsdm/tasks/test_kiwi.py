r"""Test task implementation with InSilico."""

import logging

import pytest
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from tsdm.datatools import CallableDataset, timedelta
from tsdm.encoders import FittableEncoder
from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import KiwiBenchmark
from tsdm.timeseries.pandas import PandasTSC, SplitTimeData

__logger__ = logging.getLogger(__name__)


# FIXME: broken test! this mask is incorrect!
@pytest.mark.slow
@pytest.mark.skip(reason="This test is broken.")
def test_kiwi_task() -> None:
    r"""Test the KiwiTask."""
    split_id = (0, "train")
    LOGGER = __logger__.getChild(KiwiBenchmark.__name__)
    LOGGER.info("Testing.")
    task = KiwiBenchmark()

    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.splits[split_id], PandasTSC)
    assert isinstance(task.samplers[split_id], HierarchicalSampler)
    assert isinstance(task.generators[split_id], CallableDataset)
    assert isinstance(task.dataloaders[split_id], DataLoader)
    assert isinstance(task.encoders[split_id], FittableEncoder)
    assert task.get_train_split(split_id) == split_id
    assert callable(task.collate_fns[split_id])

    # validate generator
    generator = task.generators[split_id]

    # make sample
    sampler = task.samplers[split_id]
    key = next(iter(sampler))
    sample: SplitTimeData = generator[key]
    assert isinstance(sample, SplitTimeData)

    # validate the sample
    x = sample.context_values
    y = sample.target_values
    assert y is not None
    time = sample.context_values.index
    observables: list[str] = task.observables
    covariates: list[str] = task.covariates
    targets: list[str] = task.targets
    assert set(observables) | set(covariates) | set(targets) == set(x.columns)

    td_observation = timedelta(task.observation_horizon)
    td_forecasting = timedelta(task.forecasting_horizon)
    mask_observation = time < (time.min() + td_observation)  # FIXME: mask incorrect!
    mask_forecasting = time >= (time.min() + td_observation)
    assert all(mask_observation ^ mask_forecasting), f"{key=}"
    assert all(~mask_forecasting | (time >= (time.max() - td_forecasting))), f"{key=}"
    assert x.loc[mask_observation, observables].notna().any().any(), f"{key=}"
    assert x.loc[mask_forecasting, observables].isna().all().all(), f"{key=}"
    assert x.loc[mask_observation, covariates].notna().any().any(), f"{key=}"
    assert x.loc[mask_forecasting, covariates].notna().any().any(), f"{key=}"
    assert y.loc[mask_observation, targets].isna().all().all(), f"{key=}"
    assert y.loc[mask_forecasting, targets].notna().any().any(), f"{key=}"

    dataloader = task.dataloaders[split_id]
    batch = next(iter(dataloader))
    assert isinstance(batch, list | tuple)
    sample = batch[0]
    assert isinstance(sample, tuple | Tensor)
