"""Implementation of the SIR model from epidemiology."""

__all__ = ["SIR"]

from typing import Any

from tsdm.random.generators._generators import IVP_Generator
from tsdm.types.aliases import SizeLike
from tsdm.types.variables import any_co as T_co


class SIR(IVP_Generator):
    """"""

    def get_initial_state(self, size: SizeLike = ()) -> T_co:
        pass

    def make_observations(self, sol: Any, /) -> T_co:
        pass

    def system(self, t, x):
        """



        :param t:
        :param x:
        :return:
        """
