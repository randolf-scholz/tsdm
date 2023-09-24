#!/usr/bin/env python
r"""Test PhysioNet 2012."""

import logging

from pytest import mark

from tsdm.random.generators import (
    GENERATORS,
    BouncingBall,
    DampedPendulum,
    LoktaVolterra,
)


@mark.parametrize("cls", GENERATORS.values(), ids=GENERATORS.keys())
def test_physionet_2012(cls):
    r"""Test PhysioNet 2012."""
    dataset = Physionet2012().dataset
    metadata, series = dataset["A"]

    print(metadata)
    print(series)


def __main__():
    logging.basicConfig(level=logging.INFO)
    test_physionet_2012()


if __name__ == "__main__":
    __main__()
