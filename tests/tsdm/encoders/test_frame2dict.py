r"""Test the standardizer encoder."""

from pandas import DataFrame

from tsdm.encoders import FrameAsDict


def test_frame2tensordict() -> None:
    """Make sure that the column order is preserved."""
    df = DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
        "D": [1, 2, 3],
    })
    encoder = FrameAsDict(groups={"X": ["B"], "Y": ...})
    encoder.fit(df)
    encoded = encoder.encode(df)
    X = encoded["X"].numpy()
    Y = encoded["Y"].numpy()
    assert (X == df["B"].values).all(), "X should be equal to B"
    assert (Y == df[["A", "C", "D"]].values).all(), "Y should be equal to A, C, D"
