r"""Tests for :mod:`tsdm.utils.remote`."""

from pathlib import Path
from typing import Any

import pytest
from httpx2 import Client, MockTransport, Request, Response

from tsdm.utils import remote


def test_download_disables_read_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    r"""Downloads keep reading while a slow server is making progress."""
    captured_timeout: dict[str, Any] = {}

    def handler(request: Request) -> Response:
        captured_timeout.update(request.extensions["timeout"])
        return Response(200, content=b"test payload")

    def client_factory(*_args: Any, **_kwargs: Any) -> Client:
        return Client(transport=MockTransport(handler))

    monkeypatch.setattr(remote, "Client", client_factory)
    target = tmp_path / "data"
    remote.download("https://example.com/data", target)

    assert target.read_bytes() == b"test payload"
    assert captured_timeout == {
        "connect": 10,
        "read": None,
        "write": 10,
        "pool": 10,
    }
