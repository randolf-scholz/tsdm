#!/usr/bin/env python
r"""Test the torch encoders."""

import logging

import numpy as np
import torch

from tsdm.encoders import PositionalEncoder
from tsdm.encoders.torch import PositionalEncoder as PositionalEncoder_Torch
from tsdm.encoders.torch import Time2Vec

logging.basicConfig(level=logging.NOTSET)

__logger__ = logging.getLogger(__name__)


def test_PositionalEncoder() -> None:
    """Test the PositionalEncoder."""
    __logger__.info("Testing %s.", PositionalEncoder)
    logger = __logger__.getChild(PositionalEncoder.__name__)

    scale = 1.23
    N = 10
    num_dim = 6
    t = np.sort(np.random.rand(N))
    logger.info("Start Testing")

    try:
        logger.info("Initialization")
        encoder = PositionalEncoder(num_dim, scale)
        encoder.fit(None)
    except Exception as E:
        logger.error("Initialization failed")
        raise RuntimeError from E

    try:
        logger.info("Forward")
        y = encoder.encode(t)
    except Exception as E:
        logger.error("Forward failed")
        raise RuntimeError from E

    try:
        logger.info("Inverse")
        t_inverse = encoder.decode(y)
    except Exception as E:
        logger.error("Inverse failed")
        raise RuntimeError("Failed to run PositionalEncoder inverse") from E
    else:
        assert np.allclose(t_inverse, t), "inverse failed"

    logger.info("Finished Testing")


def test_PositionalEncoder_Torch() -> None:
    """Test the PositionalEncoder_Torch."""
    __logger__.info("Testing %s.", PositionalEncoder_Torch)
    logger = __logger__.getChild(PositionalEncoder_Torch.__name__)

    scale = 1.23
    N = 10
    num_dim = 6
    t = torch.rand(N).sort().values
    logger.info("Start Testing")

    try:
        logger.info("Initialization")
        encoder = PositionalEncoder_Torch(num_dim, scale=scale)
    except Exception as E:
        logger.error("Initialization failed")
        raise RuntimeError from E

    try:
        logger.info("Forward")
        y = encoder(t)
    except Exception as E:
        logger.error("Forward failed")
        raise RuntimeError from E

    try:
        logger.info("Inverse")
        t_inverse = encoder.inverse(y)
    except Exception as E:
        logger.error("Inverse failed")
        raise RuntimeError("Failed to run PositionalEncoder inverse") from E
    else:
        assert torch.allclose(t_inverse, t), "inverse failed"

    logger.info("Finished Testing")


def test_Time2Vec() -> None:
    """Test the Time2Vec encoder."""
    __logger__.info("Testing %s.", Time2Vec)
    logger = __logger__.getChild(Time2Vec.__name__)

    N = 10
    num_dim = 6
    t = torch.rand(N).sort().values
    logger.info("Start Testing")

    try:
        logger.info("Initialization")
        encoder = Time2Vec(num_dim, "sin")
    except Exception as E:
        logger.error("Initialization failed")
        raise RuntimeError("Failed to initialize Time2Vec") from E

    try:
        logger.info("Forward")
        y = encoder(t)
    except Exception as E:
        logger.error("Forward failed")
        raise RuntimeError("Failed to run Time2Vec") from E

    try:
        logger.info("Inverse")
        t_inverse = encoder.inverse(y)
    except Exception as E:
        logger.error("Inverse failed")
        raise RuntimeError("Failed to run Time2Vec inverse") from E
    else:
        assert torch.allclose(t_inverse, t), "inverse failed"

    logger.info("Finished Testing")


def _main() -> None:
    test_PositionalEncoder()
    test_Time2Vec()


if __name__ == "__main__":
    _main()
