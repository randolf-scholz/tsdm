r"""Implementation of the kiwi task."""

__all__ = [
    # Classes
    "KiwiSampleGenerator",
    "KiwiTask",
]


from collections.abc import Callable, Hashable
from typing import NamedTuple, TypeVar

from pandas import DataFrame
from torch import Tensor
from torch import nan as NAN
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler as TorchSampler

from tsdm.datasets import KiwiDataset, TimeSeriesCollection
from tsdm.encoders import (
    BoundaryEncoder,
    BoxCoxEncoder,
    Frame2TensorDict,
    FrameEncoder,
    IdentityEncoder,
    LinearScaler,
    LogitBoxCoxEncoder,
    MinMaxScaler,
    ModularEncoder,
    Standardizer,
    TimeDeltaEncoder,
)
from tsdm.random.samplers import HierarchicalSampler, SlidingWindowSampler
from tsdm.tasks.base import Sample, TimeSeriesSampleGenerator, TimeSeriesTask
from tsdm.utils.data import folds_as_frame, folds_as_sparse_frame, folds_from_groups
from tsdm.utils.strings import repr_namedtuple
from tsdm.utils.types import KeyVar

SplitID = TypeVar("SplitID", bound=Hashable)


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.
    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

    def __repr__(self):
        return repr_namedtuple(self)


class KiwiSampleGenerator(TimeSeriesSampleGenerator):
    r"""Sample generator for the KIWI dataset."""

    def __init__(self, dataset):
        super().__init__(
            dataset,
            observables=[
                "Base",
                "DOT",
                "Glucose",
                "OD600",
                "Acetate",
                "Fluo_GFP",
                "pH",
                "Temperature",
            ],
            covariates=[
                "Cumulated_feed_volume_glucose",
                "Cumulated_feed_volume_medium",
                "InducerConcentration",
                "StirringSpeed",
                "Flow_Air",
                "Probe_Volume",
            ],
            targets=["Fluo_GFP"],
        )


class KiwiTask(TimeSeriesTask):
    r"""Task for the KIWI dataset."""
    # dataset: TimeSeriesCollection = KiwiDataset()
    observation_horizon: str = "2h"
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: str = "1h"
    r"""The number of datapoints the model should forecast."""

    def __init__(self) -> None:
        dataset = KiwiDataset()
        dataset.timeseries = dataset.timeseries.astype("float64")
        super().__init__(dataset=dataset)

    @staticmethod
    def default_metric(*, targets, predictions):
        r"""TODO: implement this."""

    def make_collate_fn(self, key: SplitID, /) -> Callable[[list[Sample]], Batch]:
        r"""TODO: implement this."""
        encoder = self.encoders[key]

        def collate_fn(samples: list[Sample]) -> Batch:

            x_vals: list[Tensor] = []
            y_vals: list[Tensor] = []
            x_time: list[Tensor] = []
            y_time: list[Tensor] = []
            x_mask: list[Tensor] = []
            y_mask: list[Tensor] = []

            for sample in samples:
                tx, x = encoder.encode(sample.inputs.x).values()
                ty, y = encoder.encode(sample.targets.y).values()
                # create a mask for looking up the target values
                x_time.append(tx)
                x_vals.append(x)
                x_mask.append(x.isfinite())

                y_time.append(ty)
                y_vals.append(y)
                y_mask.append(y.isfinite())

            return Batch(
                x_time=pad_sequence(x_time, batch_first=True).squeeze(),
                x_vals=pad_sequence(
                    x_vals, batch_first=True, padding_value=NAN
                ).squeeze(),
                x_mask=pad_sequence(x_mask, batch_first=True).squeeze(),
                y_time=pad_sequence(y_time, batch_first=True).squeeze(),
                y_vals=pad_sequence(
                    y_vals, batch_first=True, padding_value=NAN
                ).squeeze(),
                y_mask=pad_sequence(y_mask, batch_first=True).squeeze(),
            )

        return collate_fn

    def make_encoder(self, key: KeyVar, /) -> ModularEncoder:
        VF = self.dataset.value_features
        column_encoders = {}
        for col, scale, lower, upper in VF[["scale", "lower", "upper"]].itertuples():
            encoder: ModularEncoder
            match scale:
                case "percent":
                    encoder = (
                        LogitBoxCoxEncoder()
                        @ LinearScaler(lower, upper)
                        @ BoundaryEncoder(lower, upper, mode="clip")
                    )
                case "absolute":
                    encoder = BoxCoxEncoder() @ BoundaryEncoder(
                        lower, upper, mode="clip"
                    )
                case "linear":
                    encoder = IdentityEncoder()
                case _:
                    raise ValueError(f"{scale=} unknown")
            column_encoders[col] = encoder

        encoder = (
            Frame2TensorDict(
                groups={
                    "key": ["run_id", "exp_id"],
                    "T": ["measurement_time"],
                    "X": ...,
                },
                dtypes={"T": "float32", "X": "float32"},
            )
            @ Standardizer()
            @ FrameEncoder(
                column_encoders=column_encoders,
                index_encoders={
                    # "run_id": IdentityEncoder(),
                    # "exp_id": IdentityEncoder(),
                    "measurement_time": MinMaxScaler()
                    @ TimeDeltaEncoder(),
                },
            )
        )

        self.LOGGER.info("Initializing Encoder for key='%s'", key)
        train_key = self.train_partition[key]
        associated_train_split = self.splits[train_key]
        self.LOGGER.info("Fitting encoder to associated train split '%s'", train_key)
        encoder.fit(associated_train_split.timeseries)

        return encoder

    def make_sampler(self, key: KeyVar, /) -> TorchSampler:
        split: TimeSeriesCollection = self.splits[key]
        subsamplers = {
            key: SlidingWindowSampler(tsd.index, horizons=["2h", "1h"], stride="1h")
            for key, tsd in split.items()
        }
        return HierarchicalSampler(split, subsamplers, shuffle=False)  # type: ignore[return-value]

    def make_folds(self, /) -> DataFrame:
        r"""Group by RunID and color which indicates replicates."""
        md = self.dataset.metadata
        groups = md.groupby(["run_id", "color"], sort=False).ngroup()
        folds = folds_from_groups(
            groups, seed=2022, num_folds=5, train=7, valid=1, test=2
        )
        df = folds_as_frame(folds)
        return folds_as_sparse_frame(df)

    def make_generator(self, key: KeyVar, /) -> KiwiSampleGenerator:
        split = self.splits[key]
        return KiwiSampleGenerator(split)
