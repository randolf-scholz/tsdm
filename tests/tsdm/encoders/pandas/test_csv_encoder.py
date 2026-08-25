r"""Test encoders defined in `tsdm.encoders.dataframe`."""

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from tsdm.config import PROJECT
from tsdm.encoders.pandas import CSVEncoder

RESULTS_DIR = PROJECT.RESULTS_DIR[__file__]


TEST_FRAME_A = DataFrame(
    {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
        "D": [1, 2, 3],
    }
)


def test_csv_encoder() -> None:
    # initialize encoder
    encoder = CSVEncoder(RESULTS_DIR / "test.csv")
    assert not encoder.requires_fit

    # encode frame
    path = encoder.encode(TEST_FRAME_A)
    assert path.exists()

    # compare decoded frame with original
    result = encoder.decode(path)
    assert_frame_equal(TEST_FRAME_A, result)
