r"""Test the standardizer encoder."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from pandas import DataFrame

from tsdm.config import PROJECT
from tsdm.encoders import (
    BaseEncoder,
    BoundaryEncoder,
    BoxCoxEncoder,
    DateTimeEncoder,
    Encoder,
    FrameAsTensorDict,
    FrameEncoder,
    IdentityEncoder,
    LogitBoxCoxEncoder,
    MinMaxScaler,
    StandardScaler,
)
from tsdm.tasks import KiwiBenchmark

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.fixture
def encoder(scope="session"):
    # initialize the task object
    task = KiwiBenchmark()
    descr = task.dataset.timeseries_description[["kind", "lower_bound", "upper_bound"]]

    # select encoding scheme
    column_encoders: dict[str, Encoder] = {}
    for col, scale, lower, upper in descr.itertuples():
        xmin = None if pd.isna(lower) else float(lower)
        xmax = None if pd.isna(upper) else float(upper)

        match scale:
            case "percent" | "fraction":
                assert (xmin, xmax) != (None, None)
                column_encoders[col] = (
                    BoundaryEncoder(xmin, xmax, mode="clip")
                    >> MinMaxScaler(0, 1, xmin=xmin, xmax=xmax)
                    >> LogitBoxCoxEncoder()
                )
            case "absolute":
                if xmax is not None and xmax < np.inf:
                    column_encoders[col] = (
                        BoundaryEncoder(xmin, xmax, mode="clip")
                        # >> MinMaxScaler(lower, upper)
                        >> BoxCoxEncoder()
                    )
                else:
                    column_encoders[col] = (
                        BoundaryEncoder(xmin, xmax, mode="clip") >> BoxCoxEncoder()
                    )
            case "linear":
                column_encoders[col] = IdentityEncoder()
            case _:
                raise ValueError(f"{scale=} unknown")

    # construct the encoder
    encoder = (
        FrameEncoder(
            column_encoders=column_encoders,
            index_encoders={"measurement_time": DateTimeEncoder() >> MinMaxScaler()},
        )
        >> StandardScaler(axis=-1)
        >> FrameAsTensorDict(
            schema={"T": ["measurement_time"], "X": ...},
            dtypes={"T": torch.float32, "X": torch.float32},
        )
    )

    # fit encoder to the whole dataset
    ts = task.dataset.timeseries
    train_data = ts.iloc[:20_000].reset_index(
        level=["run_id", "experiment_id"], drop=True
    )
    encoder.fit(train_data)

    return encoder


@pytest.mark.slow
def test_combined_encoder(encoder, SplitID=(0, "train"), atol=1e-5, rtol=2**-12):
    r"""Test complicated combined encoder."""
    # initialize the task object
    torch.manual_seed(0)
    np.random.seed(0)  # noqa: NPY002
    task = KiwiBenchmark()

    # prepare train data
    ts = task.dataset.timeseries.iloc[:20_000]
    train_data = ts.reset_index(level=["run_id", "experiment_id"], drop=True)

    # prepare test data
    sampler = task.samplers[SplitID]
    generator = task.generators[SplitID]
    key = next(iter(sampler))
    sample = generator[key]
    test_data = sample.inputs.x

    # fit encoder to the whole dataset
    encoder.fit(train_data)

    # encode and validate
    train_encoded = encoder.encode(train_data)
    assert isinstance(train_encoded, dict)
    assert all(isinstance(value, torch.Tensor) for value in train_encoded.values())

    # check NaN-pattern and standardization
    xhat_train = DataFrame(train_encoded["X"], dtype="float32")
    assert (xhat_train.isna() == train_data.isna()).all(), "NaN pattern mismatch"
    assert np.allclose(xhat_train.mean().dropna(), 0.0, atol=atol)
    assert np.allclose(xhat_train.std(ddof=0).dropna(), 1.0, atol=atol)

    # check that decode gives back original values
    train_decoded = encoder.decode(train_encoded)
    pd.testing.assert_frame_equal(train_decoded, train_data, atol=atol, rtol=2**-12)

    # apply encoder to a single slice
    test_encoded = encoder.encode(test_data)
    assert isinstance(test_encoded, dict)
    assert all(isinstance(value, torch.Tensor) for value in test_encoded.values())

    # check NaN-pattern and standardization
    xhat_test = DataFrame(test_encoded["X"], dtype="float32")
    assert (xhat_test.isna() == test_data.isna()).all(), "NaN pattern mismatch"
    assert np.allclose(xhat_test.mean().dropna(), 0.0, atol=atol)
    assert np.allclose(xhat_test.std(ddof=0).dropna(), 1.0, atol=atol)

    # check that decoded matches with original
    test_decoded = encoder.decode(test_encoded)
    pd.testing.assert_frame_equal(test_decoded, test_data, atol=atol, rtol=rtol)


def test_bounds(encoder):
    task = KiwiBenchmark()
    descr = task.dataset.timeseries_description[["kind", "lower_bound", "upper_bound"]]

    nrows, ncols = task.dataset.timeseries.shape

    # check that decoding random values satisfies bounds
    rng_data = {
        "X": 20 * torch.randn(nrows, ncols, dtype=torch.float32),
        "T": torch.randn(nrows, 1),
    }
    rng_decoded = encoder.decode(rng_data)

    # validate bounds
    bounds = pd.concat(
        [rng_decoded.min(), rng_decoded.max()],
        axis=1,
        keys=["lower", "upper"],
    )
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


def test_serialization(encoder):
    # test_serialization
    with open(RESULT_DIR / "trained_encoder.pickle", "wb") as file:
        pickle.dump(encoder, file)

    with open(RESULT_DIR / "trained_encoder.pickle", "rb") as file:
        loaded_encoder = pickle.load(file)

    assert isinstance(loaded_encoder, BaseEncoder)
