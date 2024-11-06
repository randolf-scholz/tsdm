r"""Test the MIMIC-IV dataset import."""

import pytest

from tsdm.datasets import MIMIC_IV


@pytest.mark.slow
def test_mimic_iv() -> None:
    MIMIC_IV()
