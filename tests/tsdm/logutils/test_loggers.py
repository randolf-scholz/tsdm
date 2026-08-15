r"""Test the logutils module."""

import tempfile
from pathlib import Path

import torch
from pandas import DataFrame
from torch.utils.tensorboard.writer import SummaryWriter

import tsdm
from tsdm.config import PROJECT
from tsdm.logutils import BaseLogger, DefaultLogger, log_table
from tsdm.metrics import MSE

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


def test_base_logger() -> None:
    r"""Test the BaseLogger class."""
    # initialize logger
    logger = BaseLogger()

    # add callback
    metrics = {"MSE": MSE}
    writer = SummaryWriter(RESULT_DIR)
    cb = tsdm.logutils.MetricsCallback(metrics, writer=writer)
    assert cb.required_kwargs == {"targets", "predictions"}
    logger.add_callback("batch", cb)
    # run callbacks
    targets = torch.randn(10, 3)
    predics = torch.randn(10, 3)
    logger["batch"].callback(1, targets=targets, predictions=predics)
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


def test_log_table() -> None:
    r"""Serialize tables through the shared table-serialization utility."""
    table = DataFrame({"values": [1, 2]})

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        log_table(
            3,
            path,
            table,
            filetype="csv",
            options={"index": False},
            name="metrics",
            prefix="validation",
            postfix="final",
        )

        assert (path / "validation:metrics:final-3.csv").read_text() == "values\n1\n2\n"
