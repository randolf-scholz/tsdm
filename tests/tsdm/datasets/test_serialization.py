r"""Test Serialization Datasets."""

from pathlib import Path
from tempfile import TemporaryDirectory

import polars.testing as pl_testing

from tsdm.datasets import InSilico


def test_serialize() -> None:
    ds = InSilico()

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "insilio.zip"
        ds.serialize(path)
        ds2 = InSilico.deserialize(path)

    for key in ds:
        pl_testing.assert_frame_equal(ds[key], ds2[key])
