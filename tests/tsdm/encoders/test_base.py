"""Test basic encoder functionality."""

from tsdm.encoders import EncoderProtocol, IdentityEncoder


def test_identity_encoder():
    DEMO = IdentityEncoder() @ IdentityEncoder()
    DEMO.__repr__()
    assert isinstance(DEMO, EncoderProtocol)
