r"""Test encoders defined in `tsdm.encoders.dataframe`."""

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from tsdm.config import PROJECT
from tsdm.encoders.dataframe import CSVEncoder, DTypeConverter

RESULTS_DIR = PROJECT.RESULTS_DIR[__file__]


TEST_FRAME_A = DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9],
    "D": [1, 2, 3],
})


def test_csv_encoder() -> None:
    # initialize encoder
    encoder = CSVEncoder(RESULTS_DIR / "test.csv")
    assert not encoder.requires_fit

    # encode frame
    path = encoder._encode_impl(TEST_FRAME_A)
    assert path.exists()

    # compare decoded frame with original
    result = encoder.decode(path)
    assert_frame_equal(TEST_FRAME_A, result)


def test_type_converter() -> None:
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
