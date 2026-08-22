r"""Dataset Wrapper for the Damped Pendulum Generator."""

__all__ = [
    "TIMESERIES_SCHEMA",
    "TIMESERIES_METADATA",
    "TIMESERIES_METADATA_SCHEMA",
    "DampedPendulum_Ansari2023",
]

from typing import Literal, final

import numpy as np
import polars as pl
from scipy.stats import norm as univariate_normal
from tqdm.auto import trange

from tsdm.datasets.base import PolarsDataset
from tsdm.random import generators

type Key = Literal["timeseries", "timeseries_metadata"]

TIMESERIES_SCHEMA = {
    "sequence_id": pl.Int64,
    "time": pl.Float64,
    "x": pl.Float32,
    "y": pl.Float32,
}
TIMESERIES_METADATA_SCHEMA = {
    "variable"        : pl.String,
    "lower_bound"     : pl.Float32,
    "upper_bound"     : pl.Float32,
    "lower_inclusive" : pl.Boolean,
    "upper_inclusive" : pl.Boolean,
    "unit"            : pl.String,
    "description"     : pl.String,
}  # fmt: skip
TIMESERIES_METADATA = [
    ("sequence_id", None, None, None, None, None, "trajectory ID"),
    ("time",        None, None, None, None,  "s", "Elapsed time" ),
    ("x", -1.0, +1.0, True, True, "length", "x coordinate of the pendulum bob"),
    ("y", -1.0, +1.0, True, True, "length", "y coordinate of the pendulum bob"),
]  # fmt: skip


@final
class DampedPendulum_Ansari2023(PolarsDataset[Key]):
    r"""Dataset Wrapper for the Damped Pendulum Generator.

    Note:
        We follow the description of the Damped Pendulum in paper [1], see
        appendix C1.

    References:
        - | Neural Continuous-Discrete State Space Models
          | Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh
          | Proceedings of the 40th International Conference on Machine Learning
          | https://proceedings.mlr.press/v202/ansari23a.html
          | https://github.com/clear-nus/NCDSSM
        - | Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data
          | Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt
          | ICLR 2017
          | https://openreview.net/forum?id=HyTqHL5xg
        - | Deep Rao-Blackwellised Particle Filters for Time Series Forecasting
          | Richard Kurle, Syama Sundar Rangapuram, Emmanuel de Bézenac, Stephan Günnemann, Jan Gasthaus
          | NeurIPS 2020
          | https://proceedings.neurips.cc/paper/2020/hash/afb0b97df87090596ae7c503f60bb23f-Abstract.html
          | https://dl.acm.org/doi/10.5555/3495724.3497013
    """

    rawdata_files = []
    table_names = ["timeseries", "timeseries_metadata"]  # pyright: ignore[reportAssignmentType]
    table_schemas = {  # pyright: ignore[reportAssignmentType]
        "timeseries": TIMESERIES_SCHEMA,
        "timeseries_metadata": TIMESERIES_METADATA_SCHEMA,
    }
    table_shapes = {  # pyright: ignore[reportAssignmentType]
        "timeseries": (1_057_000, 4),
        "timeseries_metadata": (4, 7),
    }

    num_sequences = 7000
    step = 0.1
    t_min = 0.0
    t_max = 15.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.generator = generators.DampedPendulumXY(
            length=1.0,
            g=9.81,
            mass=1.0,
            gamma=0.25,
            theta0=np.pi,
            omega0=4.0,
            observation_noise_dist=univariate_normal(loc=0, scale=0.05),
            initial_state_dist=univariate_normal(loc=0, scale=1),
        )

    @staticmethod
    def clean_timeseries_metadata() -> pl.DataFrame:
        r"""Create DataFrame with metadata for the timeseries."""
        return pl.DataFrame(
            TIMESERIES_METADATA,
            schema=TIMESERIES_METADATA_SCHEMA,
            orient="row",
        )

    def clean_timeseries(self) -> pl.DataFrame:
        self.LOGGER.info("Generating data...")

        # generate time range
        t_range = np.arange(self.t_min, self.t_max + self.step / 2, self.step)
        if t_range[0] != self.t_min or t_range[-1] != self.t_max:
            raise ValueError(f"Invalid time range: {t_range[0]=}, {t_range[-1]=}")
        if not np.allclose(np.diff(t_range), self.step):
            raise ValueError(f"Invalid time step: {np.diff(t_range)=}")

        data = np.empty((self.num_sequences, len(t_range), 2), dtype=np.float32)
        for sequence_id in trange(self.num_sequences, desc="generating sequences"):
            data[sequence_id] = self.generator.rvs(t_range)

        return pl.DataFrame(
            {
                "sequence_id": np.repeat(np.arange(self.num_sequences), len(t_range)),
                "time": np.tile(t_range, self.num_sequences),
                "x": data[..., 0].ravel(),
                "y": data[..., 1].ravel(),
            }
        )
