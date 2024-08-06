r"""Tests for `tsdm.types.protocols.SupportsKwargs`."""

import pytest

from tsdm.types.protocols import SupportsKwargs


def test_supports_kwargs():
    d_int = {0: 1, 1: 2}
    with pytest.raises(NotImplementedError):
        issubclass(d_int, SupportsKwargs)  # type: ignore[arg-type]
    assert isinstance(d_int, SupportsKwargs) is False

    d_str = {"a": 1, "b": 2}
    with pytest.raises(NotImplementedError):
        issubclass(d_str, SupportsKwargs)  # type: ignore[arg-type]
    assert isinstance(d_str, SupportsKwargs) is True
