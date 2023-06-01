#!/usr/bin/env python
r"""Testing Attribute Serialization."""

import logging

import pandas as pd
import torch

from tsdm.encoders import BaseEncoder
from tsdm.models.pretrained import LinODEnet

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


def test_pretrained() -> None:
    """Test the serialization of models."""
    __logger__.info("Testing %s.", LinODEnet)

    checkpoints = LinODEnet.available_checkpoints()
    assert isinstance(checkpoints, pd.DataFrame)

    pretrained = LinODEnet.from_remote_checkpoint("2022-12-01/270")
    assert isinstance(pretrained, LinODEnet)
    assert isinstance(pretrained.components, dict)

    model = pretrained.components["LinODEnet"]
    assert isinstance(model, torch.nn.Module)

    encoder = pretrained.components["encoder"]
    assert isinstance(encoder, BaseEncoder)

    hyperparameters = pretrained.components["hparams"]
    assert isinstance(hyperparameters, dict)

    # optimizer = pretrained.components["optimizer"]
    # assert isinstance(optimizer, torch.optim.Optimizer)


def _main() -> None:
    test_pretrained()


if __name__ == "__main__":
    _main()
