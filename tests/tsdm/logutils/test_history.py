r"""Test independent evaluation and best-epoch callbacks with shared histories."""

import csv
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from importlib import import_module
from io import StringIO
from math import isfinite, isnan
from pathlib import Path
from statistics import fmean
from typing import Literal

import polars as pl
import pytest
import torch
import yaml
from polars.testing import assert_frame_equal
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.logutils import (
    CALLBACKS,
    BaseCallback,
    BestEpochTracker,
    CallbackList,
    DefaultLogger,
    EvaluationCallback,
    HParamCallback,
)

event_accumulator = import_module(
    "tensorboard.backend.event_processing.event_accumulator"
)

HISTORY_CSV = """step,train/loss,validation/accuracy,test/accuracy
10,9.0,0.50,0.45
20,8.0,0.60,0.55
30,7.0,0.55,0.52
40,6.0,0.70,0.65
50,5.0,0.70,0.99
60,4.0,0.65,0.61
70,3.0,0.80,0.75
80,2.0,0.78,0.97
90,1.0,0.85,0.81
100,0.5,0.82,0.88
"""

BEST_EPOCH_CSV = """step,train/loss,validation/accuracy,test/accuracy,best_step
10,9.0,0.50,0.45,10
20,8.0,0.60,0.55,20
30,8.0,0.60,0.55,20
40,6.0,0.70,0.65,40
50,6.0,0.70,0.65,40
60,6.0,0.70,0.65,40
70,3.0,0.80,0.75,70
80,3.0,0.80,0.75,70
90,1.0,0.85,0.81,90
100,1.0,0.85,0.81,90
"""


@pytest.mark.parametrize("batch_size", [1, 3, 10])
def test_best_epoch_matches_csv_example(batch_size: int) -> None:
    r"""Select by validation accuracy and carry every raw score from its epoch."""
    history = pl.read_csv(StringIO(HISTORY_CSV))
    expected = pl.read_csv(StringIO(BEST_EPOCH_CSV))
    tracker = BestEpochTracker(
        history, selection_column="validation/accuracy", mode="max"
    )
    for step in history["step"].to_list()[batch_size - 1 :: batch_size]:
        tracker(step)
        assert_frame_equal(
            tracker.best_history,
            expected.filter(pl.col("step") <= step),
            check_exact=True,
        )
    tracker(history["step"][-1])
    assert_frame_equal(tracker.best_history, expected, check_exact=True)


def scores(loss: float, error: float = 1.0) -> dict[str, dict[str, float]]:
    return {
        "train": {"loss": loss + 10, "error": error + 10},
        "holdout": {"loss": loss, "error": error},
        "test": {"loss": loss + 20, "error": error + 20},
    }


def frame(step: int, loss: float, error: float = 1.0, /) -> pl.DataFrame:
    values = {
        f"{split}/{metric}": value
        for split, metrics in scores(loss, error).items()
        for metric, value in metrics.items()
    }
    return pl.DataFrame(
        [{"step": step, **values}],
        schema={"step": pl.Int64, **dict.fromkeys(values, pl.Float64)},
    )


type WriterFactory = Callable[[Path], SummaryWriter]


@pytest.fixture
def writers() -> Iterator[WriterFactory]:
    r"""Close real TensorBoard writers after each test, including on failures."""
    with ExitStack() as resources:

        def create(path: Path, /) -> SummaryWriter:
            writer = SummaryWriter(log_dir=str(path))
            resources.callback(writer.close)
            return writer

        yield create


def predict(model: nn.Module, batch: Sequence[Tensor], /) -> tuple[Tensor, Tensor]:
    (data,) = batch
    return data, model(data)


def mse(*, predictions: Tensor, targets: Tensor) -> Tensor:
    return (predictions - targets).square().mean()


def mae(*, predictions: Tensor, targets: Tensor) -> Tensor:
    return (predictions - targets).abs().mean()


def make_loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)


def make_evaluation(
    writer: SummaryWriter, /, *, frequency: int = 1
) -> EvaluationCallback:
    return EvaluationCallback(
        nn.Identity(),
        predict_fn=predict,
        dataloaders=dict.fromkeys(scores(0), make_loader()),
        metrics={"loss": mse, "error": mae},
        writer=writer,
        frequency=frequency,
    )


@pytest.mark.parametrize(("window", "min_periods"), [(1, 1), (3, 1), (3, 3), (5, 1)])
@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("metric", ["loss", "error"])
def test_incremental_selection_matches_reference(
    *, window: int, min_periods: int, mode: Literal["min", "max"], metric: str
) -> None:
    r"""Match a full-prefix reference while preserving shared table identities."""
    observations = [
        (float("nan"), 9.0),
        (5.0, 7.0),
        (1.0, 1.0),
        (4.0, 2.0),
        (float("inf"), 3.0),
        (-float("inf"), 1.0),
        (2.0, float("nan")),
        (0.0, 0.0),
        (0.0, 0.0),
        (8.0, 8.0),
    ]
    batches = [frame(10 * i + 3, *pair) for i, pair in enumerate(observations)]
    history = batches[0].clear()
    tracker = BestEpochTracker(
        history,
        selection_column=f"holdout/{metric}",
        window=window,
        min_periods=min_periods,
        mode=mode,
    )
    best_history, selection_history = tracker.best_history, tracker.processed_history
    for end, batch in enumerate(batches):
        previous = best_history.clone()
        history.vstack(batch, in_place=True)
        tracker(batch["step"].item())
        assert tracker.history is history
        assert tracker.best_history is best_history
        assert tracker.processed_history is selection_history
        assert_frame_equal(best_history.head(previous.height), previous)
        assert best_history.height == selection_history.height == end + 1
        assert best_history.columns == [*history.columns, "best_step"]
        index = ("loss", "error").index(metric)
        candidates = []
        for position in range(end + 1):
            values = [
                pair[index]
                for pair in observations[max(0, position - window + 1) : position + 1]
                if isfinite(pair[index])
            ]
            if len(values) >= min_periods and isfinite(observations[position][index]):
                candidates.append((position, fmean(values)))
        if not candidates:
            assert tracker.best["best_step"].item() is None
            assert all(
                isnan(value) for value in tracker.best.drop("step", "best_step").row(0)
            )
            continue
        choose = min if mode == "min" else max
        position, criterion = choose(candidates, key=lambda item: item[1])
        assert tracker.best["step"].item() == batch["step"].item()
        assert tracker.best["best_step"].item() == batches[position]["step"].item()
        assert tracker.selection_value == pytest.approx(criterion)
        expected = batches[position].drop("step")
        assert_frame_equal(
            tracker.best.select(expected.columns), expected, check_exact=True
        )


def test_existing_table_and_exact_step_labels() -> None:
    r"""Process only through the requested label, without using it as an index."""
    history = pl.DataFrame(
        {"loss": [9.0, 1.0, 4.0], "step": [3, 13, 23], "count": [5, 7, 9]}
    )
    tracker = BestEpochTracker(history, selection_column="loss")
    assert isinstance(tracker, BaseCallback)
    assert CALLBACKS["BestEpochTracker"] is BestEpochTracker
    assert tracker.best_history.is_empty()
    tracker(13)
    assert tracker.best_history["step"].to_list() == [3, 13]
    best = tracker.best.drop("step", "best_step").row(0, named=True)
    assert best == {"loss": 1.0, "count": 7}
    tracker(23)
    assert tracker.best_history["best_step"].to_list() == [3, 13, 13]
    assert tracker.best_history.schema["count"] == pl.Int64
    before = tracker.best_history.clone()
    tracker(13)
    tracker(23)
    assert_frame_equal(tracker.best_history, before)
    with pytest.raises(KeyError, match="14"):
        tracker(14)
    assert_frame_equal(tracker.best_history, before)


def test_selection_history_and_raw_winning_row() -> None:
    r"""Inspect processed criteria while copying all raw scores from one epoch."""
    losses = [9.0, 1.0, 4.0, float("nan"), 8.0, float("inf"), 2.0]
    history = pl.concat(
        [frame(10 * i + 3, loss, float(i)) for i, loss in enumerate(losses)]
    )
    tracker = BestEpochTracker(
        history, selection_column="holdout/loss", window=3, min_periods=2
    )
    tracker(63)
    assert tracker.processed_history.columns == ["step", "holdout/loss"]
    actual_losses = tracker.processed_history["holdout/loss"].to_list()
    expected_losses = [None, 5.0, 14 / 3, 2.5, 6.0, None, 5.0]
    assert actual_losses == pytest.approx(expected_losses)
    # The low mean at the NaN epoch is inspectable but cannot make that raw row win.
    assert tracker.best["best_step"].item() == 23
    assert tracker.selection_value == pytest.approx(14 / 3)
    assert_frame_equal(
        tracker.best.drop("step", "best_step"), frame(23, 4.0, 2.0).drop("step")
    )


def test_tracker_catches_up_with_rolling_windows() -> None:
    losses = [9.0, 1.0, 4.0, 16.0, 2.0, 8.0, 0.0, 7.0, 3.0, 6.0, 5.0, 10.0]
    history = pl.concat([frame(10 * i, loss) for i, loss in enumerate(losses)])
    tracker = BestEpochTracker(history, selection_column="holdout/loss", window=3)
    tracker(110)
    assert tracker.processed_history["step"].to_list() == list(range(0, 120, 10))
    actual_losses = tracker.processed_history["holdout/loss"].to_list()
    expected_losses = [
        fmean(losses[max(0, end - 2) : end + 1]) for end in range(len(losses))
    ]
    assert actual_losses == pytest.approx(expected_losses)
    assert tracker.best_history.height == tracker.processed_history.height == 12
    previous = tracker.best_history.clone(), tracker.processed_history.clone()
    tracker(110)
    assert_frame_equal(tracker.best_history, previous[0])
    assert_frame_equal(tracker.processed_history, previous[1])


@pytest.mark.parametrize(
    "options",
    [
        {"window": 0},
        {"min_periods": 0},
        {"window": 2, "min_periods": 3},
        {"mode": "bad"},
    ],
)
def test_invalid_tracker_options(options: dict) -> None:
    arguments: dict = {"selection_column": "holdout/loss"}
    arguments.update(options)
    with pytest.raises(ValueError, match=r"window|mode|column"):
        BestEpochTracker(frame(0, 1).clear(), **arguments)


def test_extreme_step_labels() -> None:
    history = pl.DataFrame({"step": [-(2**63), 0, 2**63 - 1], "loss": [3.0, 1.0, 2.0]})
    tracker = BestEpochTracker(history, selection_column="loss")
    tracker(2**63 - 1)
    assert tracker.best["best_step"].item() == 0


def test_evaluation_without_tracking_and_default_selection(
    tmp_path: Path, writers: WriterFactory
) -> None:
    evaluation = make_evaluation(writers(tmp_path))
    history = evaluation.history
    assert set(tmp_path.glob("*.csv")) == {tmp_path / "history.csv"}
    assert pl.read_csv(tmp_path / "history.csv").is_empty()
    for step, loss in [(0, 9.0), (1, 1.0), (2, 4.0)]:
        evaluation.append_history(step, scores(loss))
    assert evaluation.history is history
    assert history.height == 3
    tracker = BestEpochTracker(history, selection_column="holdout/loss")
    tracker(2)
    assert tracker.best_history["best_step"].to_list() == [0, 1, 1]


def test_both_csv_headers_and_append_only_outputs(
    tmp_path: Path, writers: WriterFactory
) -> None:
    path = tmp_path / "new" / "run"
    evaluation = make_evaluation(writers(path))
    tracker = BestEpochTracker(
        evaluation.history, selection_column="holdout/loss", writer=evaluation.writer
    )
    files = [path / "history.csv", path / "best_history.csv"]
    headers = {file: file.read_bytes() for file in files}
    for name, table in [
        ("history", evaluation.history),
        ("best_history", tracker.best_history),
    ]:
        with (path / f"{name}.csv").open(newline="") as file:
            assert list(csv.reader(file)) == [table.columns]
    for step, loss in [(3, 1 + 2e-8), (10, 1 + 1e-8)]:
        evaluation.append_history(step, scores(loss))
        tracker(step)
        for file, prefix in headers.items():
            assert file.read_bytes().startswith(prefix)
        headers = {file: file.read_bytes() for file in files}
    for name, table in [
        ("history", evaluation.history),
        ("best_history", tracker.best_history),
    ]:
        assert_frame_equal(
            pl.read_csv(path / f"{name}.csv", schema_overrides=table.schema),
            table,
            check_exact=True,
        )
    assert tracker.best["holdout/loss"].item() == 1 + 1e-8
    evaluation.writer.flush()
    events = event_accumulator.EventAccumulator(str(path)).Reload()
    logged = events.Scalars("score:loss/test")
    assert [event.step for event in logged] == [3, 10]
    assert logged[-1].value == pytest.approx(21.0)


@pytest.mark.parametrize(
    "split", ["", "1train", "train/test", "train set", "train-set"]
)
def test_evaluation_split_names_must_be_identifiers(
    tmp_path: Path, *, split: str, writers: WriterFactory
) -> None:
    with pytest.raises(ValueError, match="identifier"):
        EvaluationCallback(
            nn.Identity(),
            predict_fn=predict,
            dataloaders={split: make_loader()},
            metrics={"loss": mse},
            writer=writers(tmp_path),
        )
    assert not (tmp_path / "history.csv").exists()


def test_wide_metric_names_and_metadata_collisions(
    tmp_path: Path, writers: WriterFactory
) -> None:
    splits = ["holdout", "test", "step", "best_step", "δοκιμή", "class"]
    metrics = ["loss/a/b", "step", 'quoted,"metric']
    evaluation = EvaluationCallback(
        nn.Identity(),
        predict_fn=predict,
        dataloaders=dict.fromkeys(splits, make_loader()),
        metrics=dict.fromkeys(metrics, mse),
        writer=writers(tmp_path),
    )
    tracker = BestEpochTracker(
        evaluation.history,
        selection_column="holdout/loss/a/b",
        writer=evaluation.writer,
    )
    for step in [10, 20]:
        values = {split: dict.fromkeys(metrics, 30.0 - step) for split in splits}
        evaluation.append_history(step, values)
        tracker(step)
    expected = {
        f"{split}/{metric}": value
        for split, metrics in values.items()
        for metric, value in metrics.items()
    }
    assert tracker.best.drop("step", "best_step").row(0, named=True) == expected
    assert tracker.best_history.columns == [*evaluation.history.columns, "best_step"]
    for name, table in [
        ("history", evaluation.history),
        ("best_history", tracker.best_history),
    ]:
        path = tmp_path / f"{name}.csv"
        with path.open(newline="") as file:
            assert next(csv.reader(file)) == table.columns
        assert_frame_equal(
            pl.read_csv(path, skip_rows=1, has_header=False, schema=table.schema), table
        )


def test_duplicate_evaluations_are_rejected(
    tmp_path: Path, writers: WriterFactory
) -> None:
    evaluation = make_evaluation(writers(tmp_path))
    evaluation(3)
    before = (tmp_path / "history.csv").read_bytes()
    for step in [3, 2]:
        with pytest.raises(ValueError, match="after 3"):
            evaluation(step)
    with pytest.raises(ValueError, match="must match"):
        evaluation.append_history(4, {"holdout": {"loss": 1.0}})
    assert evaluation.history.height == 1
    assert (tmp_path / "history.csv").read_bytes() == before


def test_tracker_failed_write_preserves_shared_tables(
    tmp_path: Path, writers: WriterFactory
) -> None:
    evaluation = make_evaluation(writers(tmp_path))
    tracker = BestEpochTracker(
        evaluation.history, selection_column="holdout/loss", writer=evaluation.writer
    )
    evaluation.append_history(3, scores(9))
    tracker(3)
    evaluation.append_history(10, scores(1))
    outputs = [tracker.processed_history, tracker.best_history]
    before = [table.clone() for table in outputs]
    csv_path = tmp_path / "best_history.csv"
    backup = csv_path.with_suffix(".backup")
    csv_path.rename(backup)
    persisted = backup.read_bytes()
    # A directory at the output path produces a real filesystem write failure.
    csv_path.mkdir()
    with pytest.raises(IsADirectoryError):
        tracker(10)
    assert tracker.processed_history is outputs[0]
    assert tracker.best_history is outputs[1]
    for actual, expected in zip(outputs, before, strict=True):
        assert_frame_equal(actual, expected)
    assert tracker.selection_value == 9.0
    assert tracker.best["best_step"].item() == 3
    assert tracker.processed_history.height == 1
    assert backup.read_bytes() == persisted

    # Once the output path is restored, retry the same step without duplicate rows.
    csv_path.rmdir()
    backup.rename(csv_path)
    tracker(10)
    assert tracker.best["best_step"].item() == 10
    assert tracker.selection_value == 1.0
    assert tracker.processed_history.height == 2
    assert_frame_equal(
        pl.read_csv(csv_path, schema_overrides=tracker.best_history.schema),
        tracker.best_history,
    )


@pytest.mark.parametrize("finalize", [False, True])
def test_callback_frequencies_and_final_reporting(
    tmp_path: Path, *, finalize: bool, writers: WriterFactory
) -> None:
    model = nn.Linear(1, 1, bias=False)
    logger = DefaultLogger(
        log_dir=tmp_path,
        results_dir=tmp_path,
        checkpoint_dir=tmp_path,
        model=model,
        predict_fn=predict,
        dataloaders=dict.fromkeys(scores(0), make_loader()),
        metrics={"loss": mse, "error": mae},
        writer=writers(tmp_path),
        selection_column="holdout/loss",
        selection_window=3,
        hparam_dict={"lr": 0.1},
    )
    evaluation = next(
        cb for cb in logger["epoch"] if isinstance(cb, EvaluationCallback)
    )
    tracker = logger.best_epoch
    assert tracker is not None
    evaluation.frequency = 2
    tracker.frequency = 4
    hparams = next(cb for cb in logger["result"] if isinstance(cb, HParamCallback))
    assert hparams.history is tracker.best_history
    callbacks = CallbackList([evaluation, tracker])
    weights = {2: 4.0, 4: 2.0, 6: 3.0}
    for step in range(1, 7):
        if step in weights:
            with torch.no_grad():
                model.weight.fill_(weights[step])
        callbacks.callback(step)
    assert evaluation.history["step"].to_list() == [2, 4, 6]
    assert tracker.processed_history["step"].to_list() == [2, 4]
    raw_history = evaluation.history.clone()
    report_step = 6 if finalize else 4
    if finalize:
        logger.log_results(report_step)
    else:
        hparams(report_step)
    assert_frame_equal(evaluation.history, raw_history)
    expected_steps = [2, 4, 6] if finalize else [2, 4]
    assert tracker.processed_history["step"].to_list() == expected_steps
    assert tracker.best["best_step"].item() == (6 if finalize else 4)
    selected_scores = (
        {"loss": 4.0, "error": 2.0} if finalize else {"loss": 1.0, "error": 1.0}
    )
    expected = {
        f"{split}/{metric}": value
        for split in scores(0)
        for metric, value in selected_scores.items()
    }
    assert tracker.best.drop("step", "best_step").row(0, named=True) == expected
    expected["best_step"] = report_step
    saved = yaml.safe_load((tmp_path / f"{report_step}.yaml").read_text())
    assert saved == expected
    assert type(saved["best_step"]) is int
    evaluation.writer.flush()
    events = event_accumulator.EventAccumulator(str(tmp_path)).Reload()
    for column, value in expected.items():
        assert events.Scalars(f"metrics:hparam/{column}")[-1].value == value


@pytest.mark.parametrize("selected", [False, True])
def test_hparam_reporting_requires_a_nonempty_history(
    tmp_path: Path, *, selected: bool, writers: WriterFactory
) -> None:
    history = frame(1, 0).clear()
    tracker = BestEpochTracker(history, selection_column="holdout/loss")
    callback = HParamCallback(
        {},
        history=tracker.best_history if selected else history,
        writer=writers(tmp_path),
    )
    with pytest.raises(ValueError, match="requires a nonempty history"):
        callback(1)
    assert not (tmp_path / "1.yaml").exists()


@pytest.mark.parametrize("selected", [False, True])
def test_hparam_reporting_allows_missing_scores(
    tmp_path: Path, *, selected: bool, writers: WriterFactory
) -> None:
    evaluation = make_evaluation(writers(tmp_path))
    tracker = BestEpochTracker(evaluation.history, selection_column="holdout/loss")
    source = tracker.best_history if selected else evaluation.history
    callback = HParamCallback({}, history=source, writer=evaluation.writer)
    evaluation.append_history(3, scores(float("nan"), float("nan")))
    tracker(3)
    callback(3)
    saved = yaml.safe_load((tmp_path / "3.yaml").read_text())
    if selected:
        assert saved.pop("best_step") is None
    assert set(saved) == set(evaluation.history.columns[1:])
    assert all(isnan(value) for value in saved.values())
    evaluation.writer.flush()
    events = event_accumulator.EventAccumulator(str(tmp_path)).Reload()
    assert "metrics:hparam/best_step" not in events.Tags()["scalars"]
    for column in saved:
        assert isnan(events.Scalars(f"metrics:hparam/{column}")[-1].value)


@pytest.mark.parametrize("report_step", [15, 35])
@pytest.mark.parametrize("selected", [False, True])
def test_hparam_reporting_selects_exact_step_and_preserves_score_types(
    tmp_path: Path, *, report_step: int, selected: bool, writers: WriterFactory
) -> None:
    history = pl.DataFrame(
        {
            "step": [5, 15, 25, 35],
            "objective": [9.0, 1.0, 4.0, 20.0],
            "quality": [100.0, 70.0, 80.0, 60.0],
            "count": [100, 200, 300, 400],
            "another metric/with/slashes": [1.0, 2.0, 3.0, 4.0],
        }
    )
    tracker = BestEpochTracker(history, selection_column="objective", window=3)
    tracker(35)
    writer = writers(tmp_path)
    source = tracker.best_history if selected else history
    callback = HParamCallback({"lr": 0.1}, history=source, writer=writer)
    assert callback.history is source
    callback(report_step)
    assert tracker.best["best_step"].item() == 25
    winning_step = 15 if report_step == 15 else 25
    expected = (
        history.filter(pl.col("step") == (winning_step if selected else report_step))
        .drop("step")
        .row(0, named=True)
    )
    if selected:
        expected["best_step"] = winning_step
    saved = yaml.safe_load((tmp_path / f"{report_step}.yaml").read_text())
    assert saved == expected
    for column, value in expected.items():
        assert type(saved[column]) is type(value)
    writer.flush()
    events = event_accumulator.EventAccumulator(str(tmp_path)).Reload()
    for column, value in expected.items():
        assert events.Scalars(f"metrics:hparam/{column}")[-1].value == value


def test_hparam_reporting_preserves_null_scores(
    tmp_path: Path, *, writers: WriterFactory
) -> None:
    history = pl.DataFrame({"step": [3], "score": [None], "best_step": [3]})
    writer = writers(tmp_path)
    callback = HParamCallback({}, history=history, writer=writer)
    callback(3)
    saved = yaml.safe_load((tmp_path / "3.yaml").read_text())
    assert saved == {"score": None, "best_step": 3}
    assert type(saved["best_step"]) is int
    writer.flush()
    events = event_accumulator.EventAccumulator(str(tmp_path)).Reload()
    assert events.Tags()["scalars"] == ["metrics:hparam/best_step"]
    assert events.Scalars("metrics:hparam/best_step")[-1].value == 3


def test_hparam_reporting_requires_an_existing_step(
    tmp_path: Path, *, writers: WriterFactory
) -> None:
    history = pl.DataFrame({"step": [10, 20], "score": [1.0, 2.0]})
    callback = HParamCallback({}, history=history, writer=writers(tmp_path))
    with pytest.raises(pl.exceptions.NoRowsReturnedError, match="returned no rows"):
        callback(15)
    assert not (tmp_path / "15.yaml").exists()


@pytest.mark.parametrize("track", [False, True])
def test_default_logger_optional_tracking(
    tmp_path: Path, *, track: bool, writers: WriterFactory
) -> None:
    loader = make_loader()
    logger = DefaultLogger(
        log_dir=tmp_path,
        results_dir=tmp_path,
        checkpoint_dir=tmp_path,
        model=nn.Identity(),
        predict_fn=predict,
        dataloaders={"holdout": loader, "test": loader},
        metrics={"mse": mse},
        writer=writers(tmp_path),
        selection_column="holdout/mse" if track else None,
        hparam_dict={"lr": 0.1},
    )
    evaluation = next(
        cb for cb in logger["epoch"] if isinstance(cb, EvaluationCallback)
    )
    assert logger.history is evaluation.history
    relevant = CallbackList(
        callback
        for callback in logger["epoch"]
        if isinstance(callback, (EvaluationCallback, BestEpochTracker))
    )
    relevant.callback(1)
    if track:
        assert logger.best_epoch is not None
        assert logger.best_epoch.history is evaluation.history
        assert logger.best_history is logger.best_epoch.best_history
        assert logger.best_epoch.best["best_step"].item() == 1
        logger.log_results(1)
    else:
        assert logger.best_epoch is None
        assert logger.best_history is None
        assert not (tmp_path / "best_history.csv").exists()
