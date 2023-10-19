"""Test that Dataloader works for non-Dataset objects."""

from torch.utils.data import DataLoader


def test_dataloader() -> None:
    class Dataset:
        def __getitem__(self, item: int, /) -> int:
            return -item

    class Sampler:
        def __iter__(self):
            return iter(range(10))

        def __len__(self):
            return 10

    dataset = Dataset()
    sampler = Sampler()
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=5)  # type: ignore[arg-type, var-annotated]

    for batch in dataloader:
        print(batch)
        assert all(x <= 0 for x in batch)
