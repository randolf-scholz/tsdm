from torch.utils.data import Dataset


class MyDatasets(Dataset[int]):
    def __getitem__(self, key: int, /) -> int:
        return 0
