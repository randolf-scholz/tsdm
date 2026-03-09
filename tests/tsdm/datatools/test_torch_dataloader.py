r"""Test that Dataloader works for non-Dataset objects."""

from collections.abc import Iterator

from torch.utils.data import DataLoader


def test_dataloader() -> None:
    r"""Check that torch is ok with protocol for dataloader."""

    class Dataset:
        def __getitem__(self, item: int, /) -> int:
            return -item

    class Sampler:
        def __iter__(self) -> Iterator[int]:
            return iter(range(10))

        def __len__(self) -> int:
            return 10

    dataset = Dataset()
    sampler = Sampler()
    dataloader = DataLoader(
        dataset,  # type: ignore[arg-type, var-annotated]  # pyright: ignore[reportArgumentType]
        sampler=sampler,
        batch_size=5,
    )

    for batch in dataloader:
        print(batch)
        assert all(x <= 0 for x in batch)
