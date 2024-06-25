r"""Test tsdm.datasets.USHCN."""

import pytest

from tsdm.datasets import USHCN


@pytest.mark.slow
def test_ushcn():
    dataset = USHCN()
    assert isinstance(dataset, USHCN)
