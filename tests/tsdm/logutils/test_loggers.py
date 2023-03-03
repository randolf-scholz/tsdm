#!/usr/bin/env python
"""Test the logutils module."""

import logging

import torch
from torch.utils.tensorboard import SummaryWriter

import tsdm
from tsdm.metrics import MSE

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


def test_BaseLogger() -> None:
    """Test the BaseLogger class."""
    # initialize logger
    logger = tsdm.logutils.BaseLogger()

    # add callback
    metrics = {"MSE": MSE}
    writer = SummaryWriter("foo")
    cb = tsdm.logutils.MetricsCallback(metrics, writer)
    logger.callbacks["batch"].append(cb)

    # run callbacks
    targets = torch.randn(10, 3)
    predics = torch.randn(10, 3)
    logger.run_callbacks(1, key="batch", targets=targets, predics=predics)


def _main() -> None:
    test_BaseLogger()


if __name__ == "__main__":
    _main()
