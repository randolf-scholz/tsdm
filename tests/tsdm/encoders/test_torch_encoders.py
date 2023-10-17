r"""Test the torch encoders."""

import logging

import numpy as np
import torch

from tsdm.encoders import PositionalEncoder
from tsdm.encoders.pytorch import (
    PositionalEncoding as PositionalEncoder_Torch,
    Time2Vec,
)

__logger__ = logging.getLogger(__name__)


def test_PositionalEncoder() -> None:
    """Test the PositionalEncoder."""
    LOGGER = __logger__.getChild(PositionalEncoder.__name__)
    LOGGER.info("Testing.")

    scale = 1.23
    N = 10
    num_dim = 6
    t = np.sort(np.random.rand(N))
    LOGGER.info("Start Testing")

    try:
        LOGGER.info("Initialization")
        encoder = PositionalEncoder(num_dim, scale)
        encoder.fit(None)
    except Exception as E:
        LOGGER.error("Initialization failed")
        raise RuntimeError from E

    try:
        LOGGER.info("Forward")
        y = encoder.encode(t)
    except Exception as E:
        LOGGER.error("Forward failed")
        raise RuntimeError from E

    try:
        LOGGER.info("Inverse")
        t_inverse = encoder.decode(y)
    except Exception as E:
        LOGGER.error("Inverse failed")
        raise RuntimeError("Failed to run PositionalEncoder inverse") from E
    assert np.allclose(t_inverse, t), "inverse failed"

    LOGGER.info("Finished Testing")


def test_PositionalEncoder_Torch() -> None:
    """Test the PositionalEncoder_Torch."""
    LOGGER = __logger__.getChild(PositionalEncoder_Torch.__name__)
    LOGGER.info("Testing.")

    scale = 1.23
    N = 10
    num_dim = 6
    t = torch.rand(N).sort().values
    LOGGER.info("Start Testing")

    try:
        LOGGER.info("Initialization")
        encoder = PositionalEncoder_Torch(num_dim=num_dim, scale=scale)
    except Exception as E:
        LOGGER.error("Initialization failed")
        raise RuntimeError from E

    try:
        LOGGER.info("Forward")
        y = encoder(t)
    except Exception as E:
        LOGGER.error("Forward failed")
        raise RuntimeError from E

    try:
        LOGGER.info("Inverse")
        t_inverse = encoder.inverse(y)
    except Exception as E:
        LOGGER.error("Inverse failed")
        raise RuntimeError("Failed to run PositionalEncoder inverse") from E
    assert torch.allclose(t_inverse, t), "inverse failed"

    LOGGER.info("Finished Testing")


def test_Time2Vec() -> None:
    """Test the Time2Vec encoder."""
    LOGGER = __logger__.getChild(Time2Vec.__name__)
    LOGGER.info("Testing.")

    N = 10
    num_dim = 6
    t = torch.rand(N).sort().values
    LOGGER.info("Start Testing")

    try:
        LOGGER.info("Initialization")
        encoder = Time2Vec(num_dim=num_dim, activation="sin")
    except Exception as E:
        LOGGER.error("Initialization failed")
        raise RuntimeError("Failed to initialize Time2Vec") from E

    try:
        LOGGER.info("Forward")
        y = encoder(t)
    except Exception as E:
        LOGGER.error("Forward failed")
        raise RuntimeError("Failed to run Time2Vec") from E

    try:
        LOGGER.info("Inverse")
        t_inverse = encoder.inverse(y)
    except Exception as E:
        LOGGER.error("Inverse failed")
        raise RuntimeError("Failed to run Time2Vec inverse") from E
    assert torch.allclose(t_inverse, t), "inverse failed"

    LOGGER.info("Finished Testing")
