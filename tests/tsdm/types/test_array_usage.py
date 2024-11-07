r"""Test Array Protocol Usage."""

from typing import assert_type

from tsdm.types.arrays import NumericalArray


def chk[Arr: NumericalArray](x: Arr) -> Arr:
    y = x + x.min()
    assert_type(y, Arr)
    return y
