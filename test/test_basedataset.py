r"""Testing of Base Dataset."""

import logging

from tsdm.datasets import InSilicoData

__logger__ = logging.getLogger(__name__)


def test_methods_attributes():
    r"""Test if all attributes are present."""
    dataset = InSilicoData()

    attrs = {
        "__dir__",
        "base_url",
        "dataset",
        "rawdata_dir",
        "dataset_dir",
        "dataset_files",
        "load",
        "download",
        "clean",
    }

    for attr in attrs:
        assert hasattr(dataset, attr)


def __main__():
    logging.basicConfig(level=logging.INFO)
    __logger__.info("Testing METHODS_ATTRIBUTES started!")
    test_methods_attributes()
    __logger__.info("Testing METHODS_ATTRIBUTES finished!")


if __name__ == "__main__":
    __main__()
