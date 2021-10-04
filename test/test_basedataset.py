r"""Testing of Base Dataset."""

import logging

from tsdm.datasets import BaseDataset

LOGGER = logging.getLogger(__name__)


def test_methods_attributes():
    r"""Test if all attributes are present."""
    attrs = {
        "__dir__",
        "url",
        "dataset",
        "rawdata_path",
        "dataset_path",
        "dataset_file",
        "load",
        "download",
        "clean",
    }

    base_attrs = set(dir(BaseDataset))
    assert attrs <= base_attrs, f"{attrs - base_attrs} missing!"

    assert hasattr(BaseDataset, "url")
    # assert hasattr(BaseDataset, 'dataset')
    assert hasattr(BaseDataset, "rawdata_path")
    assert hasattr(BaseDataset, "dataset_path")
    assert hasattr(BaseDataset, "dataset_file")
    assert hasattr(BaseDataset, "load")
    assert hasattr(BaseDataset, "download")
    assert hasattr(BaseDataset, "clean")


def __main__():
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Testing METHODS_ATTRIBUTES started!")
    test_methods_attributes()
    LOGGER.info("Testing METHODS_ATTRIBUTES finished!")


if __name__ == "__main__":
    __main__()
