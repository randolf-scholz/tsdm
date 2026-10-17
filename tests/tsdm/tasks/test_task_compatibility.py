from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from tsdm.tasks import ETT_Zhou2021, KiwiBenchmark
from tsdm.tasks.base import Task, TimeSeriesTask
from tsdm.tasks.uci.electricity_lim2021 import ElectricityLim2021
from tsdm.timeseries.pandas import TimeSeries, TimeSeriesCollection

type AnyTask = TimeSeriesTask[Any, Any, Any, Any, Any]


def make_collection(
    columns: Sequence[str], /, *, numeric_time: bool = False, kiwi: bool = False
) -> TimeSeriesCollection:
    r"""Build enough short trajectories to exercise real fold generation."""
    columns = list(columns)
    times = (
        pd.Index(np.linspace(0, 15, 31), name="time")
        if numeric_time
        else pd.timedelta_range(0, periods=31, freq="30min", name="time")
    )
    if kiwi:
        index = pd.MultiIndex.from_product(
            [range(20), [0], times], names=["run_id", "experiment_id", "elapsed_time"]
        )
        metadata = pd.DataFrame(
            {"color": ["red"] * 20}, index=index.droplevel(-1).unique()
        )
    else:
        index = pd.MultiIndex.from_product([range(20), times], names=["run_id", "time"])
        metadata = None
    data = np.arange(len(index) * len(columns), dtype=float).reshape(len(index), -1)
    descriptions = pd.DataFrame(
        {"kind": "linear", "lower_bound": 0.0, "upper_bound": 1000.0}, index=columns
    )
    return TimeSeriesCollection(
        timeseries=pd.DataFrame(1 + data / 100, index=index, columns=columns),
        timeseries_metadata=descriptions,
        static_covariates=metadata,
    )


@pytest.fixture(
    params=[
        "insilico",
        "kiwi",
        "pendulum",
        "mimic",
        "mimic_normalized",
        "ushcn",
        "ett",
        "electricity",
    ]
)
def task(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> AnyTask:
    r"""Construct every concrete task with local data and its real initializer."""
    match request.param:
        case "insilico":
            module = import_module("tsdm.tasks.kiwi.insilico")
            dataset = make_collection(
                ["Biomass", "Product", "Substrate", "Acetate", "DOTm", "Volume", "Feed"]
            )
            monkeypatch.setattr(module, "in_silico", lambda: dataset)
            return module.InSilicoTask()
        case "kiwi":
            module = import_module("tsdm.tasks.kiwi.kiwi_benchmark")
            columns = list(
                dict.fromkeys(
                    module.KiwiBenchmark.observables + module.KiwiBenchmark.covariates
                )
            )
            dataset = make_collection(columns, kiwi=True)
            monkeypatch.setattr(module, "kiwi_benchmark", lambda: dataset)
            return module.KiwiBenchmark()
        case "pendulum":
            module = import_module("tsdm.tasks.synthetic.damped_pendulum_ansari2023")
            dataset = make_collection(["x", "y"], numeric_time=True)
            monkeypatch.setattr(module, "damped_pendulum_ansari2023", lambda: dataset)
            monkeypatch.setattr(module.DampedPendulum_Ansari2023, "train_size", 12)
            monkeypatch.setattr(module.DampedPendulum_Ansari2023, "valid_size", 4)
            monkeypatch.setattr(module.DampedPendulum_Ansari2023, "test_size", 4)
            return module.DampedPendulum_Ansari2023()
        case "mimic" | "mimic_normalized":
            module = import_module("tsdm.tasks.mimic.mimic_iv_bilos2021")
            dataset = make_collection(["x", "y"], numeric_time=True)
            dataset = TimeSeriesCollection(
                timeseries=dataset.timeseries.rename_axis(
                    index=["hadm_id", "time_stamp"]
                )
            )
            monkeypatch.setattr(module, "mimic_iv_bilos2021", lambda: dataset)
            monkeypatch.setattr(module.MIMIC_IV_Bilos2021, "observation_time", 10.0)
            with pytest.warns(UserWarning, match="historical reasons"):
                return module.MIMIC_IV_Bilos2021(
                    normalize_time=request.param == "mimic_normalized"
                )
        case "ushcn":
            module = import_module("tsdm.tasks.ushcn.ushcn_debrouwer2019")
            dataset = make_collection(["x", "y"], numeric_time=True)
            monkeypatch.setattr(module, "ushcn_de_brouwer2019", lambda: dataset)
            monkeypatch.setattr(module.USHCN_DeBrouwer2019, "observation_time", 10.0)
            return module.USHCN_DeBrouwer2019()
        case "ett":
            module = import_module("tsdm.tasks.ett.ett_zhou2021")
            index = pd.date_range("2016-07-01", "2018-02-28", freq="h", name="date")
            dataset = TimeSeries(
                timeseries=pd.DataFrame(
                    {"OT": np.arange(len(index), dtype=float)}, index=index
                )
            )
            monkeypatch.setattr(module, "etth1", lambda: dataset)
            return module.ETT_Zhou2021("ETTh1")
        case "electricity":
            module = import_module("tsdm.tasks.uci.electricity_lim2021")
            index = pd.date_range("2014-01-01", "2014-09-08", freq="h", name="time")
            dataset = TimeSeries(
                timeseries=pd.DataFrame(
                    {"load": 1 + np.arange(len(index), dtype=float)}, index=index
                )
            )
            monkeypatch.setattr(module, "electricity", lambda: dataset)
            return module.ElectricityLim2021()
        case _:
            raise AssertionError(request.param)


def test_task_partitions(task: AnyTask) -> None:
    r"""All lazy components share the split keys and partitions contain datasets."""
    assert isinstance(task, Task)
    keys = set(task.split_keys)
    assert len(task.split_keys) == len(keys)
    for mapping in (
        task.partitions,
        task.generators,
        task.samplers,
        task.collate_fns,
        task.dataloaders,
    ):
        assert set(mapping) == keys
    for key in keys:
        partition = task.partitions[key]
        assert isinstance(partition, type(task.dataset))
        assert len(partition) > 0
        assert partition.timeseries.index.isin(task.dataset.timeseries.index).all()
        label = key[-1] if isinstance(key, tuple) else key
        assert task.is_train_split(key) is (label in {"train", "trainval", "joint"})
    predictions = torch.ones(2, 3, 2)
    assert float(task.error_fn(predictions, predictions)) == pytest.approx(0.0)
    assert float(task.error_fn(predictions + 1, predictions)) > 0.0
    if isinstance(task.dataset, TimeSeriesCollection):
        for fold in {key[0] for key in task.split_keys}:
            train, valid, test = (
                set(task.partitions[fold, label])
                for label in ("train", "valid", "test")
            )
            assert train.isdisjoint(valid)
            assert train.isdisjoint(test)
            assert valid.isdisjoint(test)
            assert train | valid | test == set(task.dataset)


@pytest.mark.parametrize("label", ["train", "valid", "test"])
def test_task_batch(task: AnyTask, label: str) -> None:
    r"""Exercise partition, sampler, generator and collate together for each role."""
    key = next(
        key
        for key in task.split_keys
        if (key[-1] if isinstance(key, tuple) else key) == label
    )
    loader = task.make_dataloader(key, batch_size=2, drop_last=False, pin_memory=False)
    assert isinstance(loader, DataLoader)
    batch = next(iter(loader))
    assert batch is not None
    assert task.samplers[key].shuffle is task.is_train_split(key)


def test_custom_collate_and_loader_defaults(task: AnyTask) -> None:
    r"""Explicit loader arguments take precedence over task-specific defaults."""
    key = next(iter(task.split_keys))
    loader = task.make_dataloader(
        key, batch_size=2, drop_last=False, collate_fn=len, pin_memory=False
    )
    assert isinstance(loader, DataLoader)
    assert next(iter(loader)) == 2
    assert loader.batch_size == 2
    assert loader.drop_last is False
    if isinstance(task, ETT_Zhou2021 | KiwiBenchmark):
        keys: list[Any] = list(task.split_keys)
        for key in keys:
            expected = (
                task.DEFAULT_TRAIN_BATCH_SIZE
                if task.is_train_split(key)
                else task.DEFAULT_EVAL_BATCH_SIZE
            )
            cached_loader = task.dataloaders[key]
            assert isinstance(cached_loader, DataLoader)
            assert cached_loader.batch_size == expected


@pytest.mark.parametrize("task", ["kiwi", "ett", "electricity"], indirect=True)
def test_training_statistics(task: AnyTask) -> None:
    r"""Evaluation partitions use statistics fitted exclusively on training data."""
    match task:
        case KiwiBenchmark():
            train = task.partitions[0, "train"].timeseries
            valid = task.partitions[0, "valid"].timeseries
            encoder = task.encoders[0, "valid"]
            encoded = encoder.encode(valid.droplevel(["run_id", "experiment_id"]))["X"]
            expected = (valid - train.mean()) / train.std(ddof=0)
            torch.testing.assert_close(
                encoded, torch.tensor(expected.to_numpy(), dtype=encoded.dtype)
            )
        case ETT_Zhou2021():
            train = task.preprocessor.encode(task.partitions["train"].timeseries)
            test = task.preprocessor.encode(task.partitions["test"].timeseries)
            assert train["OT"].mean() == pytest.approx(0.0, abs=1e-6)
            assert test["OT"].mean() > 1.0
        case ElectricityLim2021():
            expected = task.partitions["train"].timeseries.mean()
            pd.testing.assert_series_equal(task.encoder.mean, expected)
        case _:
            raise AssertionError(type(task))


@pytest.mark.parametrize("task", ["kiwi"], indirect=True)
def test_kiwi_configuration_is_per_instance(task: KiwiBenchmark) -> None:
    r"""Overriding one task's fold configuration does not change future tasks."""
    customized = type(task)(fold_kwargs={"num_folds": 2})
    default = type(task)()
    assert len(customized.split_keys) == 6
    assert len(default.split_keys) == 15
    assert task.fold_kwargs["num_folds"] == 5
