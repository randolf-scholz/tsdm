import polars as pl
import pytest

from tsdm.datasets import DampedPendulum_Ansari2023


@pytest.mark.manual
def test_damped_pendulum_ansari2023_preprocessing() -> None:
    DampedPendulum_Ansari2023.reset_dataset_files(force=True)
    ds = DampedPendulum_Ansari2023()

    for key in DampedPendulum_Ansari2023.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == DampedPendulum_Ansari2023.table_schemas[key]
