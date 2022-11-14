#!/usr/bin/env python
r"""Test task implementation with InSilicoData."""

import logging

import pandas as pd
import pytest
import torch.utils.data
from pandas import DataFrame, MultiIndex
from torch import Tensor
from torch.utils.data import DataLoader

from tsdm.datasets import TimeSeriesCollection
from tsdm.encoders import BaseEncoder
from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import KiwiSampleGenerator, KiwiTask
from tsdm.tasks.base import Sample

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@pytest.mark.slow
def test_kiwi_task(SplitID=(0, "train")):
    r"""Test the KiwiTask."""
    LOGGER = __logger__.getChild(KiwiTask.__name__)
    LOGGER.info("Testing.")
    task = KiwiTask()

    # reveal_type(task)
    # reveal_type(task.folds)
    # reveal_type(task.index)
    # reveal_type(task.splits)
    # reveal_type(task.samplers)
    # reveal_type(task.generators)
    # reveal_type(task.dataloaders)
    # reveal_type(task.encoders)
    # reveal_type(task.train_partition)
    # reveal_type(task.collate_fns)

    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.index, MultiIndex)
    assert isinstance(task.splits[SplitID], TimeSeriesCollection)
    assert isinstance(task.samplers[SplitID], HierarchicalSampler)
    assert isinstance(task.generators[SplitID], KiwiSampleGenerator)
    assert isinstance(task.dataloaders[SplitID], DataLoader)
    assert isinstance(task.encoders[SplitID], BaseEncoder)
    assert isinstance(task.train_partition, dict)
    assert callable(task.collate_fns[SplitID])

    sampler = task.samplers[SplitID]
    key = next(iter(sampler))
    generator = task.generators[SplitID]
    assert isinstance(generator, torch.utils.data.Dataset)
    sample: Sample = generator[key]
    assert isinstance(sample, tuple)

    # validate the sample
    x = sample.inputs.x
    y = sample.targets.y
    time = sample.inputs.x.index
    observables: list[str] = generator.observables
    covariates: list[str] = generator.covariates
    targets: list[str] = generator.targets
    td_observation = pd.Timedelta(task.observation_horizon)
    td_forecasting = pd.Timedelta(task.forecasting_horizon)
    mask_observation = time < time.min() + td_observation
    mask_forecasting = time >= time.max() - td_forecasting

    assert all(mask_observation ^ mask_forecasting)
    assert set(observables) | set(covariates) | set(targets) == set(x.columns)
    assert x.loc[mask_observation, observables].notna().any().any()
    assert x.loc[mask_forecasting, observables].isna().all().all()
    assert x.loc[mask_observation, covariates].notna().any().any()
    assert x.loc[mask_forecasting, covariates].notna().any().any()
    assert y.loc[mask_observation, targets].isna().all().all()
    assert y.loc[mask_forecasting, targets].notna().any().any()

    dataloader = task.dataloaders[SplitID]
    batch = next(iter(dataloader))
    assert isinstance(batch, list | tuple)
    sample = batch[0]
    assert isinstance(sample, tuple | Tensor)


def _main() -> None:
    test_kiwi_task()


if __name__ == "__main__":
    _main()
