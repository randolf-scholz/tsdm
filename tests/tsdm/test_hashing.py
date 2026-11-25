r"""Tests for file and archive hashing."""

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from tsdm.hashing import hash_file, hash_zip_contents


@pytest.mark.parametrize("block_size", [0, -1])
def test_hash_file_rejects_nonpositive_block_size(
    tmp_path: Path, block_size: int
) -> None:
    r"""An invalid block size raises before opening the file."""
    with pytest.raises(ValueError, match="block_size must be positive"):
        hash_file(tmp_path / "missing.bin", block_size=block_size)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_hash_zip_contents_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    r"""An invalid chunk size raises before opening the archive."""
    with (
        BytesIO() as archive,
        pytest.raises(ValueError, match="chunk_size must be positive"),
    ):
        hash_zip_contents(archive, chunk_size=chunk_size)


def test_hash_zip_contents_rejects_duplicate_member_names() -> None:
    r"""A ZIP with two entries sharing a name has no content digest."""
    with BytesIO() as stream:
        with ZipFile(stream, "w") as archive:
            archive.writestr("data.csv", b"first")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("data.csv", b"second")

        with pytest.raises(ValueError, match=r"Duplicate ZIP member name: 'data\.csv'"):
            hash_zip_contents(stream)


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
