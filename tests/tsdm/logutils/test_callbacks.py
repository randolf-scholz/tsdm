from tsdm.logutils.callbacks import BaseCallback, Callback


def check_upcasting() -> None:
    r"""Check that BaseCallback can be upcast to Callback."""

    def _upcast(arg: BaseCallback, /) -> Callback:
        return arg
