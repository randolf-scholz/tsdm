#!/usr/bin/env python
r"""Testing Attribute Serialization."""

import logging

import torch

from tsdm.encoders import BaseEncoder
from tsdm.models.pretrained import LinODEnet

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


def test_pretrained():
    """Test the serialization of models."""
    __logger__.info("Testing %s.", LinODEnet)

    pretrained = LinODEnet()

    model = pretrained["model"]
    assert isinstance(model, torch.nn.Module)

    encoder = pretrained["encoder"]
    assert isinstance(encoder, BaseEncoder)

    hyperparameters = pretrained["hyperparameters"]
    assert isinstance(hyperparameters, dict)


def _main() -> None:
    test_pretrained()


if __name__ == "__main__":
    _main()
