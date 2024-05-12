r"""Test IO Protocol types."""

from io import IOBase
from tempfile import TemporaryFile

from typing_extensions import IO, get_protocol_members

from tsdm.types.protocols import ReadBuffer, WriteBuffer

WRITE_TYPES = {"pandas": WriteBuffer, "typing": IO}
READ_TYPES = {"pandas": ReadBuffer, "io": IOBase, "typing": IO}


def test_write_types():
    shared_attrs = set.intersection(*(set(dir(s)) for s in WRITE_TYPES.values()))

    with TemporaryFile("w", encoding="utf8") as file:
        shared_attrs &= set(dir(file))

    # remove dunder methods
    shared_attrs -= set(dir(object))
    protocol_members = get_protocol_members(WriteBuffer)
    assert shared_attrs == protocol_members
