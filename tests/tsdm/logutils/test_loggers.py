"""Test the logutils module."""

import torch
from torch.utils.tensorboard import SummaryWriter

import tsdm
from tsdm.config import PROJECT
from tsdm.metrics import MSE

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


def test_base_logger() -> None:
    """Test the BaseLogger class."""
    # initialize logger
    logger = tsdm.logutils.BaseLogger()

    # add callback
    metrics = {"MSE": MSE}
    writer = SummaryWriter(RESULT_DIR)
    cb = tsdm.logutils.MetricsCallback(metrics, writer)
    assert cb.required_kwargs == {"targets", "predics"}
    logger.add_callback("batch", cb)
    assert logger.combined_kwargs("batch") == {"targets", "predics"}
    # run callbacks
    targets = torch.randn(10, 3)
    predics = torch.randn(10, 3)
    logger.callback("batch", 1, targets=targets, predics=predics)
