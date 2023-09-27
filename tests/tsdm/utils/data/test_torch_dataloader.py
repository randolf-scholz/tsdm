#!/usr/bin/env python
"""Test that Dataloader works for non-Dataset objects."""

import logging

from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


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


def _main() -> None:
    test_dataloader()


if __name__ == "__main__":
    _main()
