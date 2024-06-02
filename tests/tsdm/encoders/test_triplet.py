r"""Test encoders defined in `tsdm.encoders.dataframe`."""

import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal

from tsdm.config import PROJECT
from tsdm.encoders.dataframe import TripletDecoder, TripletEncoder

RESULTS_DIR = PROJECT.RESULTS_DIR[__file__]


TEST_FRAMES = data_frames(
    columns=[
        column("A", dtype=float),
        column("B", dtype=float),
        column("C", dtype=float),
        column("D", dtype=float),
    ],
)
COLUMNS = Index(["A", "B", "C", "D"], name="variable")

SPARSE_SCHEMA = {
    "A": pd.SparseDtype(bool),
    "B": pd.SparseDtype(bool),
    "C": pd.SparseDtype(bool),
    "D": pd.SparseDtype(bool),
}
TALL_SCHEMA = {"value": float, "variable": pd.CategoricalDtype(COLUMNS)}
WIDE_SCHEMA = {"A": float, "B": float, "C": float, "D": float}

# region train-frame simple index ------------------------------------------------------
TRAIN_FRAME_WIDE = DataFrame(
    {
        "A": [1   , None, 3   ],
        "B": [4   , 5   , None],
        "C": [None, 8   , 9   ],
        "D": [1   , None, None],
    },
    columns=COLUMNS,
    index=Index([1, 3, 7], name="time"),
).astype(WIDE_SCHEMA)  # fmt: skip

TRAIN_FRAME_TALL = DataFrame(
    {
        "variable": ["A", "B", "D", "B", "C", "A", "C"],
        "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
    },
    index=Index([1, 1, 1, 3, 3, 7, 7], name="time"),
).astype(TALL_SCHEMA)

TRAIN_FRAME_SPARSE = DataFrame(
    {
        "A": [True, False, False, False, False, True, False],
        "B": [False, True, False, True, False, False, False],
        "C": [False, False, False, False, True, False, True],
        "D": [False, False, True, False, False, False, False],
        "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
    },
    index=Index([1, 1, 1, 3, 3, 7, 7], name="time"),
).astype(SPARSE_SCHEMA)
# endregion train-frame simple index ---------------------------------------------------


# region test-frame simple index -------------------------------------------------------
TEST_FRAME_WIDE = DataFrame(
    {
        "A": [8   , None, 0   , None],
        "B": [5   , 2   , None, None],
        "C": [None, 3   , 1   , 1   ],
        "D": [None, None, None, None],
    },
    columns=COLUMNS,
    index=Index([0, 2, 7, 9], name="time"),
).astype(WIDE_SCHEMA)  # fmt: skip

TEST_FRAME_TALL = DataFrame(
    {
        "variable": ["A", "B", "B", "C", "A", "C", "C"],
        "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
    },
    index=Index([0, 0, 2, 2, 7, 7, 9], name="time"),
).astype(TALL_SCHEMA)

TEST_FRAME_SPARSE = DataFrame(
    {
        "A": [True, False, False, False, True, False, False],
        "B": [False, True, True, False, False, False, False],
        "C": [False, False, False, True, False, True, True],
        "D": [False, False, False, False, False, False, False],
        "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
    },
    index=Index([0, 0, 2, 2, 7, 7, 9], name="time"),
).astype(SPARSE_SCHEMA)
# endregion test-frame simple index ----------------------------------------------------


# region train-frame multi-index -------------------------------------------------------
TRAIN_MINDEX_WIDE = DataFrame(
    {
        "A": [1   , None, 3   ],
        "B": [4   , 5   , None],
        "C": [None, 8   , 9   ],
        "D": [1   , None, None],
    },
    columns=COLUMNS,
    index=MultiIndex.from_tuples([(1.0, 1), (1.2, 2), (2.7, 1)], names=["time", "id"]),
).astype(WIDE_SCHEMA)  # fmt: skip

TRAIN_MINDEX_TALL = DataFrame(
    {
        "variable": ["A", "B", "D", "B", "C", "A", "C"],
        "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
    },
    index=MultiIndex.from_tuples(
        [(1.0, 1), (1.0, 1), (1.0, 1), (1.2, 2), (1.2, 2), (2.7, 1), (2.7, 1)],
        names=["time", "id"],
    ),
).astype(TALL_SCHEMA)

TRAIN_MINDEX_SPARSE = DataFrame(
    {
        "A": [True, False, False, False, False, True, False],
        "B": [False, True, False, True, False, False, False],
        "C": [False, False, False, False, True, False, True],
        "D": [False, False, True, False, False, False, False],
        "value": [1.0, 4.0, 1.0, 5.0, 8.0, 3.0, 9.0],
    },
    index=MultiIndex.from_tuples(
        [(1.0, 1), (1.0, 1), (1.0, 1), (1.2, 2), (1.2, 2), (2.7, 1), (2.7, 1)],
        names=["time", "id"],
    ),
).astype(SPARSE_SCHEMA)
# endregion train-frame multi-index ----------------------------------------------------

TEST_MINDEX_WIDE = DataFrame(
    {
        "A": [8   , None, 0   , None],
        "B": [5   , 2   , None, None],
        "C": [None, 3   , 1   , 1   ],
        "D": [None, None, None, None],
    },
    columns=COLUMNS,
    index=MultiIndex.from_tuples([(0.0, 1), (2.0, 2), (7.0, 1), (9.0, 1)], names=["time", "id"]),
).astype(WIDE_SCHEMA)  # fmt: skip

TEST_MINDEX_TALL = DataFrame(
    {
        "variable": ["A", "B", "B", "C", "A", "C", "C"],
        "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
    },
    index=MultiIndex.from_tuples(
        [(0.0, 1), (0.0, 1), (2.0, 2), (2.0, 2), (7.0, 1), (7.0, 1), (9.0, 1)],
        names=["time", "id"],
    ),
).astype(TALL_SCHEMA)

TEST_MINDEX_SPARSE = DataFrame(
    {
        "A": [True, False, False, False, True, False, False],
        "B": [False, True, True, False, False, False, False],
        "C": [False, False, False, True, False, True, True],
        "D": [False, False, False, False, False, False, False],
        "value": [8.0, 5.0, 2.0, 3.0, 0.0, 1.0, 1.0],
    },
    index=MultiIndex.from_tuples(
        [(0.0, 1), (0.0, 1), (2.0, 2), (2.0, 2), (7.0, 1), (7.0, 1), (9.0, 1)],
        names=["time", "id"],
    ),
).astype(SPARSE_SCHEMA)
# endregion test-frame multi-index -----------------------------------------------------


@pytest.mark.parametrize(
    ("train_data", "test_data", "expected", "options"),
    [
        (TRAIN_FRAME_WIDE,  TRAIN_FRAME_WIDE,  TRAIN_FRAME_TALL,    {}),
        (TRAIN_FRAME_WIDE,  TEST_FRAME_WIDE,   TEST_FRAME_TALL,     {}),
        (TRAIN_FRAME_WIDE,  TRAIN_FRAME_WIDE,  TRAIN_FRAME_SPARSE,  {"sparse": True}),
        (TRAIN_FRAME_WIDE,  TEST_FRAME_WIDE,   TEST_FRAME_SPARSE,   {"sparse": True}),
        (TRAIN_MINDEX_WIDE, TRAIN_MINDEX_WIDE, TRAIN_MINDEX_TALL,   {}),
        (TRAIN_MINDEX_WIDE, TEST_MINDEX_WIDE,  TEST_MINDEX_TALL,    {}),
        (TRAIN_MINDEX_WIDE, TRAIN_MINDEX_WIDE, TRAIN_MINDEX_SPARSE, {"sparse": True}),
        (TRAIN_MINDEX_WIDE, TEST_MINDEX_WIDE,  TEST_MINDEX_SPARSE,  {"sparse": True}),
    ],
)  # fmt: skip
def test_triplet_encoder(
    *,
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


@pytest.mark.parametrize(
    ("train_data", "test_data", "expected", "options"),
    [
        (TRAIN_FRAME_TALL,    TRAIN_FRAME_TALL,    TRAIN_FRAME_WIDE,  {}),
        (TRAIN_FRAME_TALL,    TEST_FRAME_TALL,     TEST_FRAME_WIDE,   {}),
        (TRAIN_MINDEX_SPARSE, TRAIN_FRAME_SPARSE,  TRAIN_FRAME_WIDE,  {"sparse": True}),
        (TRAIN_MINDEX_SPARSE, TEST_FRAME_SPARSE,   TEST_FRAME_WIDE,   {"sparse": True}),
        (TRAIN_MINDEX_TALL,   TRAIN_MINDEX_TALL,   TRAIN_MINDEX_WIDE, {}),
        (TRAIN_MINDEX_TALL,   TEST_MINDEX_TALL,    TEST_MINDEX_WIDE,  {}),
        (TRAIN_MINDEX_SPARSE, TRAIN_MINDEX_SPARSE, TRAIN_MINDEX_WIDE, {"sparse": True}),
        (TRAIN_MINDEX_SPARSE, TEST_MINDEX_SPARSE,  TEST_MINDEX_WIDE,  {"sparse": True}),
    ],
)  # fmt: skip
def test_triplet_decoder(
    *,
    train_data: DataFrame,
    test_data: DataFrame,
    expected: DataFrame,
    options: dict,
) -> None:
    r"""Test TripletDecoder."""
    # initialize encoder
    encoder = TripletDecoder(**options)

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


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@given(train_data=TEST_FRAMES, test_data=TEST_FRAMES)
def test_triplet_encoder_hypothesis(
    *, train_data: DataFrame, test_data: DataFrame, sparse: bool
) -> None:
    r"""Test TripletEncoder with hypothesis."""
    # clean the generated data by dropping all NaN rows
    train_data = train_data.dropna(how="all")
    test_data = test_data.dropna(how="all")

    # initialize encoder
    encoder = TripletEncoder(sparse=sparse)
    encoder.fit(train_data)

    # compare decoded with original test data
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)
    assert_frame_equal(test_data, decoded)
