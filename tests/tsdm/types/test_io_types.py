"""Test IO Protocol types."""

from io import IOBase
from typing import IO

from pandas._typing import ReadBuffer, WriteBuffer

WRITE_TYPES = {"pandas": WriteBuffer, "typing": IO}
READ_TYPES = {"pandas": ReadBuffer, "io": IOBase, "typing": IO}


def test_write_types():
    shared_attrs = set.intersection(*(set(dir(s)) for s in WRITE_TYPES.values()))

    with open("foo", "w", encoding="utf8") as file:
        shared_attrs &= set(dir(file))

    print(shared_attrs - set(dir(object)))  # {'seek', 'tell', 'flush', 'seekable'}
