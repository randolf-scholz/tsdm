from tsdm.datasets import Traffic


def test_traffic() -> None:
    ds = Traffic(initialize=False)
    ds.clean(force=True)
