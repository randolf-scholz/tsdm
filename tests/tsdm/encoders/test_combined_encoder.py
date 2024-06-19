r"""Test the standardizer encoder."""

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
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
from tsdm.utils import timedelta

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


# @pytest.mark.xfail(reason="https://github.com/pandas-dev/pandas/issues/56409")
@pytest.mark.slow
def test_combined_encoder(SplitID=(0, "train"), atol=1e-5, rtol=2**-12):
    r"""Test complicated combined encoder."""
    torch.manual_seed(0)
    np.random.seed(0)  # noqa: NPY002
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

    encoder = (
        FrameEncoder(
            column_encoders=column_encoders,
            index_encoders={"measurement_time": DateTimeEncoder() >> MinMaxScaler()},
        )
        >> StandardScaler(axis=-1)
        >> FrameAsTensorDict(
            schema={
                "key": ["run_id", "experiment_id"],
                "T": ["measurement_time"],
                "X": ...,
            },
            dtypes={"T": "float32", "X": "float32"},
            # encode_index=True,
        )
    )

    encoder.fit(ts)  # fit encoder to the whole dataset

    # encode and decode
    encoded = encoder.encode(ts)

    # check normalization
    xhat = DataFrame(encoded["X"], dtype="float32")
    assert np.allclose(xhat.mean().dropna(), 0.0, atol=atol)
    assert np.allclose(xhat.std(ddof=0).dropna(), 1.0, atol=atol)

    # check that decode gives back original values
    decoded = encoder.decode(encoded)
    pd.testing.assert_frame_equal(decoded, ts, atol=atol, rtol=2**-12)

    # apply encoder to a single slice
    encoded = encoder.encode(inputs)
    xhat = DataFrame(encoded["X"], dtype="float32")
    assert (xhat.isna().values == inputs.isna().values).all(), "NaN pattern mismatch"

    # check that decoded matches with original
    decoded = encoder.decode(encoded)
    pd.testing.assert_frame_equal(
        decoded.reset_index(drop=True),
        inputs.reset_index(drop=True),
        atol=atol,
        rtol=rtol,
    )

    # manually compare index (broken for pyarrow 14.0.1)
    pa_version = tuple(map(int, pa.__version__.split(".")))
    if pa_version >= (15, 0, 0):  # FIXME: https://github.com/apache/arrow/issues/39156
        reference_index = inputs.index
        decoded_index = decoded.index
        freq = timedelta("1us")
        assert decoded_index.notna().all()
        assert decoded_index.shape == reference_index.shape
        assert decoded_index.dtype == reference_index.dtype
        assert (decoded_index - reference_index).notna().all()
        r = abs(decoded_index - reference_index) / freq
        assert r.notna().all()
        q = abs(reference_index - reference_index[0]) / freq
        assert q.notna().all()
        assert (r <= (1e-3 * q + atol)).all()

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
