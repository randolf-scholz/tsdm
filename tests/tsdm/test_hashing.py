r"""Tests for file and archive hashing."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from tsdm.hashing import hash_zip_contents


def test_hash_zip_contents_legacy_digest() -> None:
    r"""The legacy digest ignores member order, timestamps, and compression."""
    expected = "zip:C8BBF09515240CED8F26E6719690973F97957F608496B8AE4623C6E38F69FD2E"
    members = {"b.txt": b"b", "a.txt": b"a"}

    with BytesIO() as original, BytesIO() as repacked:
        with ZipFile(original, "w") as archive:
            for name, contents in members.items():
                archive.writestr(ZipInfo(name), contents, compress_type=ZIP_STORED)
        with ZipFile(repacked, "w") as archive:
            for name in reversed(members):
                info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                archive.writestr(info, members[name], compress_type=ZIP_DEFLATED)

        assert str(hash_zip_contents(original)) == expected
        assert str(hash_zip_contents(repacked)) == expected
