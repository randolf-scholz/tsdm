r"""Tests for file and archive hashing."""

from hashlib import sha3_256, sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from tsdm.hashing import Hash, hash_file, hash_zip_contents


def test_hash_key_matches_equivalent_tuple() -> None:
    r"""A tuple can look up a dictionary entry keyed by an equal Hash."""
    key = Hash("abc", "sha256")
    checksums: dict[tuple[str, str | None], str] = {key: "payload"}
    assert checksums[("abc", "sha256")] == "payload"


def test_hash_file_restores_stream_position() -> None:
    r"""A caller-owned stream is hashed from the beginning and stays usable."""
    payload = b"test payload"
    expected = Hash(sha256(payload).hexdigest(), "sha256")

    with BytesIO(payload) as stream:
        stream.seek(4)
        actual = hash_file(stream)
        assert actual == expected
        assert stream.tell() == 4


def test_hash_file_uses_selected_algorithm(tmp_path: Path) -> None:
    r"""A path is hashed with the supplied algorithm."""
    payload = b"test payload"
    path = tmp_path / "data.bin"
    path.write_bytes(payload)
    expected = Hash(sha3_256(payload).hexdigest(), "sha3_256")
    assert hash_file(path, algorithm="sha3_256") == expected


def test_hash_zip_contents_rejects_duplicates_before_reading_members() -> None:
    r"""Duplicate names are reported before a corrupt member can be read."""
    with BytesIO() as stream:
        with ZipFile(stream, "w") as archive:
            archive.writestr("data.csv", b"first")
            archive.writestr("a.csv", b"intact member", compress_type=ZIP_STORED)
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("data.csv", b"second")
        corrupted = stream.getvalue().replace(b"intact member", b"broken member", 1)

    with (
        BytesIO(corrupted) as stream,
        pytest.raises(ValueError, match=r"Duplicate ZIP member names in .*'data\.csv'"),
    ):
        hash_zip_contents(stream)


def test_hash_zip_contents_is_stable_across_repacks() -> None:
    r"""The digest ignores member order, timestamps, and compression."""
    expected = "sha256:c50236f189987f04e1701dd0e4929f6ff1a71df0739cf1ca54a82dba56a1fa4a"
    members = {"b.txt": b"b", "café.txt": b"a"}

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


def test_hash_zip_contents_reports_selected_algorithm() -> None:
    r"""The result names the hash function selected for the archive."""
    with BytesIO() as stream:
        with ZipFile(stream, "w") as archive:
            archive.writestr("data.csv", b"payload")
        actual = hash_zip_contents(stream, algorithm="sha3_256")

    assert actual.hash_algorithm == "sha3_256"
