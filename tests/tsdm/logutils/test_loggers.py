r"""Test the logutils module."""

from functools import partial
from pathlib import Path

import pytest
import torch
from pandas import DataFrame
from torch.utils.tensorboard.writer import SummaryWriter

import tsdm
from tests.test_utils import get_results_dir
from tsdm.logutils import BaseLogger, DefaultLogger, log_table

RESULT_DIR = get_results_dir(__file__)


def test_base_logger() -> None:
    r"""Test the BaseLogger class."""
    # initialize logger
    logger = BaseLogger()

    # add callback
    metrics = {"MAE": lambda predictions, targets: (predictions - targets).abs().mean()}
    writer = SummaryWriter(RESULT_DIR)
    cb = tsdm.logutils.MetricsCallback(metrics, writer=writer)
    assert cb.required_kwargs == {"targets", "predictions"}
    logger.add_callback("batch", cb)
    # run callbacks
    targets = torch.randn(10, 3)
    predictions = torch.randn(10, 3)
    logger["batch"].callback(1, predictions=predictions, targets=targets)
    print(logger)


def test_default_logger() -> None:
    r"""Test the DefaultLogger class."""
    # initialize logger
    logger = DefaultLogger(
        log_dir=RESULT_DIR / "logs",
        results_dir=RESULT_DIR / "results",
        checkpoint_dir=RESULT_DIR / "checkpoints",
    )

    assert isinstance(logger, BaseLogger)

    logger["batch"].callback(1)
    logger["epoch"].callback(1)
    logger["results"].callback(1)
    print(logger)


def test_log_table(tmp_path: Path) -> None:
    r"""Serialize a pandas table with options bound to the serializer."""
    table = DataFrame({"values": [1, 2]})

    log_table(
        3,
        tmp_path,
        table,
        serializer=partial(DataFrame.to_csv, index=False),
        name="metrics.csv",
        prefix="validation",
        postfix="final",
    )

    assert (tmp_path / "validation:metrics:final-3.csv").read_text() == "values\n1\n2\n"


@pytest.mark.parametrize(
    ("name", "filename"),
    [
        ("metrics.csv.gz", "metrics-3.csv.gz"),
        ("metrics", "metrics-3"),
        ("", "3"),
    ],
)
def test_log_table_filename(tmp_path: Path, name: str, filename: str) -> None:
    r"""Preserve filename suffixes and support arbitrary serializer return values."""

    def serialize(table: str, path: Path, /) -> int:
        return path.write_text(table)

    with SummaryWriter(tmp_path) as writer:
        log_table(3, writer, "values\n1\n2\n", serializer=serialize, name=name)

    assert (tmp_path / filename).read_text() == "values\n1\n2\n"
