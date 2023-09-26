"""Dataset Wrapper for the Damped Pendulum Generator."""

__all__ = ["DampedPendulum"]

from typing import Any, final

import numpy as np
from scipy.stats import norm as univariate_normal

from tsdm.datasets.base import BaseDataset
from tsdm.random import generators


@final
class DampedPendulum(BaseDataset):
    """Dataset Wrapper for the Damped Pendulum Generator."""

    rawdata_files = []
    num_sequences = 7000
    step = 0.1
    t_min = 0.0
    t_max = 15.0

    @property
    def generator(self) -> generators.DampedPendulum:
        return generators.DampedPendulum(
            length=1.0,
            g=9.81,
            mass=1.0,
            gamma=0.25,
            theta0=np.pi,
            omega0=4.0,
            observation_noise=univariate_normal(loc=0, scale=-0.05),
            parameter_noise=univariate_normal(loc=0, scale=1),
        )

    def clean(self) -> None:
        self.LOGGER.info("Generating data...")

        # generate time range
        t_range = np.arange(self.t_min, self.t_max + self.step / 2, self.step)
        assert t_range[0] == self.t_min
        assert t_range[-1] == self.t_max
        assert all(np.diff(t_range) == self.step)

    def load(self, *, initializing: bool = False) -> Any:
        raise NotImplementedError
