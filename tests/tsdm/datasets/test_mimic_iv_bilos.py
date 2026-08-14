r"""Manual regression tests for the Bilos et al. MIMIC-IV preprocessing."""

import pytest
from polars.testing import assert_frame_equal

from tsdm.datasets.mimic_iv import (
    MIMIC_IV_Bilos2021,
    MIMIC_IV_Bilos2021_FromPreprocessed,
)


@pytest.mark.manual
def test_raw_timeseries_matches_bilos_export() -> None:
    r"""Check that the Polars pipeline recreates ``full_dataset.csv``.

    This test requires both the original Bilos export and the parquet source
    tables produced by :class:`MIMIC_IV`, so it is deliberately manual.
    """
    reference_dataset = MIMIC_IV_Bilos2021_FromPreprocessed(initialize=False)
    implementation_dataset = MIMIC_IV_Bilos2021(initialize=False)

    assert_frame_equal(
        reference_dataset.raw_timeseries,
        implementation_dataset.raw_timeseries,
        check_exact=False,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

    assert_frame_equal(
        reference_dataset.timeseries,
        implementation_dataset.timeseries,
        check_exact=False,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
