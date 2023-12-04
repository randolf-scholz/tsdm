r"""Test the standardizer encoder."""

import pickle

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pytest import mark

from tsdm.config import PROJECT
from tsdm.encoders import (
    BaseEncoder,
    BoundaryEncoder,
    BoxCoxEncoder,
    DateTimeEncoder,
    Encoder,
    FrameAsDict,
    FrameEncoder,
    IdentityEncoder,
    LogitBoxCoxEncoder,
    MinMaxScaler,
    StandardScaler,
)
from tsdm.tasks import KiwiBenchmark

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@mark.slow
def test_combined_encoder(SplitID=(0, "train")):
    r"""Test complicated combined encoder."""
    torch.manual_seed(0)
    np.random.seed(0)
    task = KiwiBenchmark()
    ts = task.dataset.timeseries.iloc[:20_000]  # use first 20_000 values only
    descr = task.dataset.timeseries_description[["kind", "lower_bound", "upper_bound"]]
    sampler = task.samplers[SplitID]
    generator = task.generators[SplitID]
    key = next(iter(sampler))
    sample = generator[key]
    inputs = sample.inputs.x

    # Construct the encoder
    column_encoders: dict[str, Encoder] = {}
    for col, scale, lower, upper in descr.itertuples():
        if pd.isna(upper):
            upper = None
        if pd.isna(lower):
            lower = None
        match scale:
            case "percent" | "fraction":
                column_encoders[col] = (
                    LogitBoxCoxEncoder()
                    @ MinMaxScaler(0, 1, xmin=lower, xmax=upper)
                    @ BoundaryEncoder(lower, upper, mode="clip")
                )
            case "absolute":
                if upper is not None and upper < np.inf:
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
        @ StandardScaler(axis=-1)
        @ FrameEncoder(
            column_encoders=column_encoders,
            index_encoders={"measurement_time": MinMaxScaler() @ DateTimeEncoder()},
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
    encoded = encoder.encode(inputs)
    xhat = DataFrame(encoded["X"])
    assert (xhat.isna().values == inputs.isna().values).all(), "NaN pattern mismatch"

    # check that decoded matches with original
    decoded = encoder.decode(encoded)
    MAD = (decoded - inputs).abs().mean().mean()
    assert (decoded.isna().values == inputs.isna().values).all(), "NaN pattern mismatch"
    assert MAD < 1e-3, "Large deviations from original values"

    # check that decoding random values satisfies bounds
    rng_data = torch.randn_like(encoded["X"])
    encoded["X"] = 20 * rng_data  # large stdv. to ensure that bounds are violated
    decoded = encoder.decode(encoded)
    bounds = pd.concat([decoded.min(), decoded.max()], axis=1, keys=["lower", "upper"])
    for col, lower, upper in bounds.itertuples():
        match scale := descr.loc[col, "kind"]:
            case "percent":
                assert lower == 0, f"Lower bound violated {lower=}"
                assert upper == 100, f"Upper bound violated {upper=}"
            case "fraction":
                assert lower == 0, f"Lower bound violated {lower=}"
                assert upper == 1, f"Upper bound violated {upper=}"
            case "absolute":
                assert lower == 0, f"Lower bound violated {lower=}"
            case "linear":
                pass
            case _:
                raise ValueError(f"{scale=} unknown")

    # test_serialization
    with open(RESULT_DIR / "trained_encoder.pickle", "wb") as file:
        pickle.dump(encoder, file)

    with open(RESULT_DIR / "trained_encoder.pickle", "rb") as file:
        loaded_encoder = pickle.load(file)

    assert isinstance(loaded_encoder, BaseEncoder)
