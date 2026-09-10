from typing import Never, reveal_type

from experimental.numerical_types import BooleanArray
from experimental.tests.fixtures import ARRAYS1D


class TestInspection:
    def test_covariance(self) -> None:
        def _bool[T](x: BooleanArray[T]) -> BooleanArray[Never]:
            return x

    def inspect_bool_array(self) -> None:
        def _view[T](_: BooleanArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(ARRAYS1D.NP.BOOL))
        reveal_type(_view(ARRAYS1D.PT.BOOL))
        reveal_type(_view(ARRAYS1D.PD_NP.BOOL))
        reveal_type(_view(ARRAYS1D.PD_PA.BOOL))
        reveal_type(_view(ARRAYS1D.PL.BOOL))
        # fmt: on

    reveal_type(BooleanArray.__or__)
