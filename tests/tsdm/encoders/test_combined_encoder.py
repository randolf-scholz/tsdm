#!/usr/bin/env python
r"""Test the standardizer encoder."""

import logging
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from pandas import DataFrame

from tsdm.encoders import (
    BaseEncoder,
    BoundaryEncoder,
    BoxCoxEncoder,
    FastFrameEncoder,
    FrameAsDict,
    IdentityEncoder,
    LinearScaler,
    LogitBoxCoxEncoder,
    MinMaxScaler,
    StandardScaler,
    TimeDeltaEncoder,
)
from tsdm.tasks import KiwiTask

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
def test_combined_encoder(SplitID=(0, "train")):
    r"""Test complicated combined encoder."""
    task = KiwiTask()
    ts = task.dataset.timeseries.iloc[:20_000]  # use first 20_000 values only
    VF = task.dataset.timeseries_description
    sampler = task.samplers[SplitID]
    generator = task.generators[SplitID]
    key = next(iter(sampler))
    sample = generator[key]
    x = sample.inputs.x

    # Construct the encoder
    column_encoders: dict[str, BaseEncoder] = {}
    for col, scale, lower, upper in VF[["scale", "lower", "upper"]].itertuples():
        match scale:
            case "percent":
                column_encoders[col] = (
                    LogitBoxCoxEncoder()
                    @ MinMaxScaler(0, 1, xmin=lower, xmax=upper)
                    @ BoundaryEncoder(lower, upper, mode="clip")
                )
            case "absolute":
                if upper < np.inf:
                    column_encoders[col] = (
                        BoxCoxEncoder()
                        # @ MinMaxScaler(lower, upper)
                        @ BoundaryEncoder(lower, upper, mode="clip")
                    )
                else:
                    column_encoders[col] = BoxCoxEncoder() @ BoundaryEncoder(
                        lower, upper, mode="clip"
                    )
            case "linear":
                column_encoders[col] = IdentityEncoder()
            case _:
                raise ValueError(f"{scale=} unknown")

    encoder = (
        FrameAsDict(
            groups={
                "key": ["run_id", "experiment_id"],
                "T": ["measurement_time"],
                "X": ...,
            },
            dtypes={"T": "float32", "X": "float32"},
            encode_index=True,
        )
        @ StandardScaler()
        @ FastFrameEncoder(
            column_encoders=column_encoders,
            index_encoders={"measurement_time": MinMaxScaler() @ TimeDeltaEncoder()},
        )
    )

    encoder.fit(ts)  # fit encoder to the whole dataset
    encoded = encoder.encode(ts)
    decoded = encoder.decode(encoded)
    MAD = (decoded - ts).abs().mean().mean()
    assert all(decoded.isna() == ts.isna()), "NaN pattern mismatch"
    assert MAD < 1e-3, "Large deviations from original values"
    # check that the encoded values are within the bounds

    # apply encoder to a single slice
    encoded = encoder.encode(x)
    xhat = DataFrame(encoded["X"])
    assert (xhat.isna().values == x.isna().values).all(), "NaN pattern mismatch"

    # check that decoded matches with original
    decoded = encoder.decode(encoded)
    MAD = (decoded - x).abs().mean().mean()
    assert all(decoded.isna() == x.isna()), "NaN pattern mismatch"
    assert MAD < 1e-3, "Large deviations from original values"

    # check that decoding random values satisfies bounds
    rng_data = torch.randn_like(encoded["X"])
    encoded["X"] = (
        20 * rng_data
    )  # large std. dev. to ensure that the bounds are violated
    decoded = encoder.decode(encoded)
    bounds = pd.concat([decoded.min(), decoded.max()], axis=1, keys=["lower", "upper"])
    for col, lower, upper in bounds.itertuples():
        match VF.loc[col, "scale"]:
            case "percent":
                assert lower == 0, "Lower bound violated"
                assert upper == 100, "Upper bound violated"
            case "absolute":
                assert lower == 0, "Lower bound violated"

    # test_serialization
    with open("trained_encoder.pickle", "wb") as file:
        pickle.dump(encoder, file)

    with open("trained_encoder.pickle", "rb") as file:
        loaded_encoder = pickle.load(file)

    assert isinstance(loaded_encoder, BaseEncoder)


def _main() -> None:
    ...


if __name__ == "__main__":
    _main()

# encoder = (
#    FrameAsDict(
#        groups={
#            "key": ["run_id", "experiment_id"],
#            "T": ["measurement_time"],
#            "X": ...,
#        },
#        dtypes={"T": "float32", "X": "float32"},
#        encode_index=True,
#    )
#    @ Standardizer()
#    @ FrameEncoder(
#        column_encoders=column_encoders,
#        index_encoders={
#            "measurement_time": MinMaxScaler() @ TimeDeltaEncoder(),
#        },
#    )
# )
# encoder.fit(ts)  # fit encoder to the whole dataset
# encoded = encoder.encode(ts)
# decoded = encoder.decode(encoded)
#
