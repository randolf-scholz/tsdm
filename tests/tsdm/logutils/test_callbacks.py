from pathlib import Path

import polars as pl

from tsdm.logutils.callbacks import BaseCallback, Callback, TableCallback


def check_upcasting() -> None:
    r"""Check that BaseCallback can be upcast to Callback."""

    def _upcast(arg: BaseCallback, /) -> Callback:
        return arg


def test_table_callback(tmp_path: Path) -> None:
    r"""Log the current table using the supplied serializer and callback frequency."""
    callback = TableCallback(
        pl.DataFrame({"values": [1, 2]}),
        writer=tmp_path,
        serializer=pl.DataFrame.write_parquet,
        name="metrics.parquet",
        prefix="validation",
        postfix="final",
        frequency=2,
    )

    callback.callback(1)
    assert not list(tmp_path.iterdir())

    callback.callback(2)
    assert pl.read_parquet(tmp_path / "validation:metrics:final-2.parquet").equals(
        callback.table
    )

    callback.table = pl.DataFrame({"values": [3, 4]})
    callback.callback(4)
    assert pl.read_parquet(tmp_path / "validation:metrics:final-4.parquet").equals(
        callback.table
    )
