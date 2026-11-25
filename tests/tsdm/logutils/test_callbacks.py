from pathlib import Path

import polars as pl

from tsdm.logutils.callbacks import BaseCallback, Callback, CallbackList, TableCallback


def check_upcasting() -> None:
    r"""Check that BaseCallback can be upcast to Callback."""

    def _upcast(arg: BaseCallback, /) -> Callback:
        return arg


def test_callback_list_required_kwargs() -> None:
    r"""Collect the unique keyword arguments required by the current callbacks."""

    class StateCallback(Callback):
        def __init__(self, *required_kwargs: str) -> None:
            self._required_kwargs = set(required_kwargs)

        @property
        def required_kwargs(self) -> set[str]:
            return self._required_kwargs

        def callback(self, _step: int, /, **_: object) -> None:
            pass

    callbacks = CallbackList(
        [
            StateCallback("targets", "predictions"),
            StateCallback("predictions", "weights"),
        ]
    )
    assert callbacks.required_kwargs == {"targets", "predictions", "weights"}

    callbacks.pop()
    assert callbacks.required_kwargs == {"targets", "predictions"}

    callbacks.clear()
    assert callbacks.required_kwargs == set()


def test_table_callback(tmp_path: Path) -> None:
    r"""Log the current table using the supplied serializer and callback frequency."""
    callback = TableCallback(
        pl.DataFrame({"values": [1, 2]}),
        writer=tmp_path,
        serializer=pl.DataFrame.write_parquet,
        identifier="validation:metrics:final.parquet",
        frequency=2,
    )

    callback.callback(1)
    assert not list(tmp_path.iterdir())

    callback.callback(2)
    table = pl.read_parquet(tmp_path / "validation:metrics:final-2.parquet")
    assert table.equals(callback.table)

    callback.table = pl.DataFrame({"values": [3, 4]})
    callback.callback(4)
    table = pl.read_parquet(tmp_path / "validation:metrics:final-4.parquet")
    assert table.equals(callback.table)
