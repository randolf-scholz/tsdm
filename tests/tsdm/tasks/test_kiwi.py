#!/usr/bin/env python
r"""Test task implementation with InSilicoData."""

import logging

import pandas as pd
import pytest
from pandas import DataFrame, MultiIndex
from torch.utils.data import DataLoader

from tsdm.datasets import TimeSeriesCollection
from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import KiwiSampleGenerator, KiwiTask
from tsdm.tasks.base import Sample

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@pytest.mark.slow
def test_kiwi_task():
    r"""Test the KiwiTask."""
    LOGGER = __logger__.getChild(KiwiTask.__name__)
    LOGGER.info("Testing.")

    task = KiwiTask()
    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.index, MultiIndex)
    assert isinstance(task.splits[0, "train"], TimeSeriesCollection)
    assert isinstance(task.samplers[0, "train"], HierarchicalSampler)
    assert isinstance(task.generators[0, "train"], KiwiSampleGenerator)
    assert isinstance(task.dataloaders[0, "train"], DataLoader)
    assert isinstance(task.train_partition, dict)

    sampler = task.samplers[0, "train"]
    key = next(iter(sampler))
    generator: KiwiSampleGenerator = task.generators[0, "train"]
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

    dataloader = task.dataloaders[0, "train"]
    batch = next(iter(dataloader))
    assert isinstance(batch, list)
    assert isinstance(batch[0], tuple)


def _main() -> None:
    test_kiwi_task()


if __name__ == "__main__":
    _main()
