r"""#TODO add module summary line.

#TODO add module description.
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def __call__(self):
        """Subclasses should be allowed to have arbitrary signatures."""

    @abstractmethod
    def download(self) -> None:
        """Subclasses must have **exactly** this signature."""


class Model(BaseModel):
    def __call__(self, x):
        pass

    def download(self, url=None) -> None:
        pass
