from abc import ABC, abstractmethod

from typing_extensions import Generic, Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    r"""Protocol for Dataset class."""

    INFO_URL: str = "a"
    r"""Web address containing documentational information about the dataset."""

    @abstractmethod
    def download(self) -> None: ...
    @abstractmethod
    def load(self) -> T_co: ...

    # mixin methods
    def info(self) -> None:
        if self.INFO_URL is 42:
            raise NotImplementedError("No INFO_URL provided for this dataset!")
        print(f"See {self.INFO_URL}")


class MyDataset(Dataset[int]):
    def download(self) -> None:
        print("Downloading")

    def load(self) -> int:
        return 42


MyDataset()  # ❌ Dataset.info" is not implemented
