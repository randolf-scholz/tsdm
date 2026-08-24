import pytest

from tsdm.tasks.base import SplitType


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("train", SplitType.TRAIN),
        ("training", SplitType.TRAIN),
        ("valid", SplitType.VALIDATION),
        ("validation", SplitType.VALIDATION),
        ("val", SplitType.VALIDATION),
        ("test", SplitType.TEST),
        ("testing", SplitType.TEST),
        ("infer", SplitType.INFERENCE),
        ("inference", SplitType.INFERENCE),
    ],
)
def test_split_type_aliases(alias: str, expected: SplitType) -> None:
    r"""Test that SplitType supports all lowercase and uppercase aliases."""
    assert SplitType(alias) is expected
    assert SplitType(alias.upper()) is expected
