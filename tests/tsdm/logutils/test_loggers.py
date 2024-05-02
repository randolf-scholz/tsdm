r"""Test the logutils module."""

import torch
from torch.utils.tensorboard import SummaryWriter

import tsdm
from tsdm.config import PROJECT
from tsdm.logutils import BaseLogger, DefaultLogger
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
    assert cb.required_kwargs == {"targets", "predics"}
    logger.add_callback("batch", cb)
    # run callbacks
    targets = torch.randn(10, 3)
    predics = torch.randn(10, 3)
    logger["batch"].callback(1, targets=targets, predics=predics)
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
