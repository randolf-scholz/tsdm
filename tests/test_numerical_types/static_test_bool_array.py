from typing import Never, reveal_type

from numerical_types.arrays import BooleanArray

from .fixtures import SERIES


class TestInspection:
    def test_covariance(self) -> None:
        def _bool[T](x: BooleanArray[T]) -> BooleanArray[Never]:
            return x

    def inspect_bool_array(self) -> None:
        def _view[T](_: BooleanArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(SERIES.NP.BOOL))
        reveal_type(_view(SERIES.PT.BOOL))
        reveal_type(_view(SERIES.PD_NP.BOOL))
        reveal_type(_view(SERIES.PD_PA.BOOL))
        reveal_type(_view(SERIES.PL.BOOL))
        # fmt: on

    reveal_type(BooleanArray.__or__)
