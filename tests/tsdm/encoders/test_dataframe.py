r"""Test encoders defined in `tsdm.encoders.dataframe`."""

import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal

from tsdm.config import PROJECT
from tsdm.encoders.dataframe import CSVEncoder, DTypeConverter, TripletEncoder

RESULTS_DIR = PROJECT.RESULTS_DIR[__file__]


TEST_FRAMES = data_frames(
    columns=[
        column("A", dtype=float),
        column("B", dtype=float),
        column("C", dtype=float),
        column("D", dtype=float),
    ],
)

SPARSE_SCHEMA = {
    "A": pd.SparseDtype(bool),
    "B": pd.SparseDtype(bool),
    "C": pd.SparseDtype(bool),
    "D": pd.SparseDtype(bool),
}

TEST_FRAME_A = DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9],
    "D": [1, 2, 3],
})

TRAIN_FRAME_SPARSE = DataFrame(
    {
        "A": [1   , None, 3   ],
        "B": [4   , 5   , None],
        "C": [None, 8   , 9   ],
        "D": [1   , None, None],
    },
    index=Index([1, 3, 7], name="time"),
    dtype=float,
)  # fmt: skip

TEST_FRAME_SPARSE = DataFrame(
    {
        "A": [8   , None, 0   , None],
        "B": [5   , 2   , None, None],
        "C": [None, 3   , 1   , 1   ],
        "D": [None, None, None, None],
    },
    index=Index([0, 2, 7, 9], name="time"),
    dtype=float,
)  # fmt: skip

TRAIN_FRAME_SPARSE_MULTIINDEX = DataFrame(
    {
        "A": [1   , None, 3   ],
        "B": [4   , 5   , None],
        "C": [None, 8   , 9   ],
        "D": [1   , None, None],
    },
    index=MultiIndex.from_tuples([(1.0, 1), (1.2, 2), (2.7, 1)],names=["time", "id"]),
    dtype=float,
)  # fmt: skip

TEST_FRAME_SPARSE_MULTIINDEX = DataFrame(
    {
        "A": [8   , None, 0   , None],
        "B": [5   , 2   , None, None],
        "C": [None, 3   , 1   , 1   ],
        "D": [None, None, None, None],
    },
    index=MultiIndex.from_tuples([(0.0, 1), (2.0, 2), (7.0, 1), (9.0, 1)],names=["time", "id"]),
    dtype=float,
)  # fmt: skip


def test_csv_encoder():
    # initialize encoder
    encoder = CSVEncoder(RESULTS_DIR / "test.csv")
    assert not encoder.requires_fit

    # encode frame
    path = encoder.encode(TEST_FRAME_A)
    assert path.exists()

    # compare decoded frame with original
    result = encoder.decode(path)
    assert_frame_equal(TEST_FRAME_A, result)


def test_type_converter():
    # initialize encoder
    encoder = DTypeConverter({
        "A": "duration[ns][pyarrow]",
        "B": "Int64",
        "C": "Float64",
        "D": "Float64",
    })

    # fit on the test data
    assert encoder.requires_fit
    encoder.fit(TEST_FRAME_A)
    assert encoder.is_fitted

    # compare encoded frame with expected
    encoded = encoder.encode(TEST_FRAME_A)
    assert encoded.dtypes[0] == "duration[ns][pyarrow]"
    assert encoded.dtypes[1] == "Int64"
    assert encoded.dtypes[2] == "Float64"
    assert encoded.dtypes[3] == "Float64"

    # compare decoded frame with original
    decoded = encoder.decode(encoded)
    assert_frame_equal(TEST_FRAME_A, decoded)


TRIPLET_ENCODER_TEST_CASES = [
    (
        TRAIN_FRAME_SPARSE,
        TRAIN_FRAME_SPARSE,
        DataFrame(
            {
                "variable": ["A", "B", "D", "B", "C", "A", "C"],
                "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
            },
            index=Index([1, 1, 1, 3, 3, 7, 7], name="time"),
        ).astype({"variable": pd.CategoricalDtype(["A", "B", "C", "D"])}),
        {},
    ),
    (
        TRAIN_FRAME_SPARSE,
        TEST_FRAME_SPARSE,
        DataFrame(
            {
                "variable": ["A", "B", "B", "C", "A", "C", "C"],
                "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
            },
            index=Index([0, 0, 2, 2, 7, 7, 9], name="time"),
        ).astype({"variable": pd.CategoricalDtype(["A", "B", "C", "D"])}),
        {},
    ),
    (
        TRAIN_FRAME_SPARSE,
        TRAIN_FRAME_SPARSE,
        DataFrame(
            {
                "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
                "A": [True, False, False, False, False, True, False],
                "B": [False, True, False, True, False, False, False],
                "C": [False, False, False, False, True, False, True],
                "D": [False, False, True, False, False, False, False],
            },
            index=Index([1, 1, 1, 3, 3, 7, 7], name="time"),
        ).astype(SPARSE_SCHEMA),
        {"sparse": True},
    ),
    (
        TRAIN_FRAME_SPARSE,
        TEST_FRAME_SPARSE,
        DataFrame(
            {
                "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
                "A": [True, False, False, False, True, False, False],
                "B": [False, True, True, False, False, False, False],
                "C": [False, False, False, True, False, True, True],
                "D": [False, False, False, False, False, False, False],
            },
            index=Index([0, 0, 2, 2, 7, 7, 9], name="time"),
        ).astype(SPARSE_SCHEMA),
        {"sparse": True},
    ),
    (
        TRAIN_FRAME_SPARSE_MULTIINDEX,
        TRAIN_FRAME_SPARSE_MULTIINDEX,
        DataFrame(
            {
                "variable": ["A", "B", "D", "B", "C", "A", "C"],
                "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
            },
            index=MultiIndex.from_tuples(
                [(1.0, 1), (1.0, 1), (1.0, 1), (1.2, 2), (1.2, 2), (2.7, 1), (2.7, 1)],
                names=["time", "id"],
            ),
        ).astype({"variable": pd.CategoricalDtype(["A", "B", "C", "D"])}),
        {},
    ),
    (
        TRAIN_FRAME_SPARSE_MULTIINDEX,
        TEST_FRAME_SPARSE_MULTIINDEX,
        DataFrame(
            {
                "variable": ["A", "B", "B", "C", "A", "C", "C"],
                "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
            },
            index=MultiIndex.from_tuples(
                [(0.0, 1), (0.0, 1), (2.0, 2), (2.0, 2), (7.0, 1), (7.0, 1), (9.0, 1)],
                names=["time", "id"],
            ),
        ).astype({"variable": pd.CategoricalDtype(["A", "B", "C", "D"])}),
        {},
    ),
    (
        TRAIN_FRAME_SPARSE_MULTIINDEX,
        TRAIN_FRAME_SPARSE_MULTIINDEX,
        DataFrame(
            {
                "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
                "A": [True, False, False, False, False, True, False],
                "B": [False, True, False, True, False, False, False],
                "C": [False, False, False, False, True, False, True],
                "D": [False, False, True, False, False, False, False],
            },
            index=MultiIndex.from_tuples(
                [(1.0, 1), (1.0, 1), (1.0, 1), (1.2, 2), (1.2, 2), (2.7, 1), (2.7, 1)],
                names=["time", "id"],
            ),
        ).astype(SPARSE_SCHEMA),
        {"sparse": True},
    ),
    (
        TRAIN_FRAME_SPARSE_MULTIINDEX,
        TEST_FRAME_SPARSE_MULTIINDEX,
        DataFrame(
            {
                "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
                "A": [True, False, False, False, True, False, False],
                "B": [False, True, True, False, False, False, False],
                "C": [False, False, False, True, False, True, True],
                "D": [False, False, False, False, False, False, False],
            },
            index=MultiIndex.from_tuples(
                [(0.0, 1), (0.0, 1), (2.0, 2), (2.0, 2), (7.0, 1), (7.0, 1), (9.0, 1)],
                names=["time", "id"],
            ),
        ).astype(SPARSE_SCHEMA),
        {"sparse": True},
    ),
]


@pytest.mark.parametrize(
    ("train_data", "test_data", "expected", "options"), TRIPLET_ENCODER_TEST_CASES
)
def test_triplet_encoder(
    train_data: DataFrame,
    test_data: DataFrame,
    expected: DataFrame,
    options: dict,
) -> None:
    r"""Test TripletEncoder."""
    # initialize encoder
    encoder = TripletEncoder(**options)

    # fit on the training data
    assert encoder.requires_fit
    encoder.fit(train_data)
    assert encoder.is_fitted

    # compare encoded test data with expected
    encoded = encoder.encode(test_data)
    assert_frame_equal(encoded, expected)

    # compare decoded with original test data
    decoded = encoder.decode(encoded)
    assert_frame_equal(decoded, test_data)


@given(train_data=TEST_FRAMES, test_data=TEST_FRAMES)
def test_triplet_encoder_hypothesis(train_data, test_data):
    encoder = TripletEncoder()
    encoder.fit(train_data)
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)
    assert_frame_equal(test_data, decoded)
