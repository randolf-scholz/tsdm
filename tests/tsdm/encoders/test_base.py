r"""Test basic encoder functionality."""

from tsdm.encoders import EncoderProtocol, IdentityEncoder


def test_identity_encoder():
    DEMO = IdentityEncoder() @ IdentityEncoder()
    repr(DEMO)
    assert isinstance(DEMO, EncoderProtocol)
