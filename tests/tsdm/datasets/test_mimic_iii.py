r"""Test the MIMIC-III dataset import."""

import pytest

from tsdm.datasets import MIMIC_III


@pytest.mark.slow
def test_mimic_iii_scholz() -> None:
    MIMIC_III()
