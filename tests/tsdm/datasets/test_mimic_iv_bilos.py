r"""Manual regression tests for the Bilos et al. MIMIC-IV preprocessing."""

import pytest
from polars.testing import assert_frame_equal

from tsdm.datasets.mimic_iv import MIMIC_IV_Bilos2021


@pytest.mark.manual
def test_matches_from_processed() -> None:
    r"""Check that the Polars pipeline recreates ``full_dataset.csv``.

    This test requires both the original Bilos export and the parquet source
    tables produced by :class:`MIMIC_IV`, so it is deliberately manual.
    """
    MIMIC_IV_Bilos2021.reset_dataset_files(force=True)
    reference_dataset = MIMIC_IV_Bilos2021.from_processed()
    MIMIC_IV_Bilos2021.reset_dataset_files(force=True)
    implementation_dataset = MIMIC_IV_Bilos2021()

    assert_frame_equal(
        implementation_dataset.raw_timeseries,
        reference_dataset.raw_timeseries,
        check_exact=False,
        rel_tol=2**-19,  # ≈ 1.9e-6
        abs_tol=2**-19,  # ≈ 1.9e-6
    )

    assert_frame_equal(
        implementation_dataset.timeseries,
        reference_dataset.timeseries,
        check_exact=False,
        rel_tol=2**-19,  # ≈ 1.9e-6
        abs_tol=2**-19,  # ≈ 1.9e-6
    )
