from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class A(Generic[T]):
    x: None | T = None

    def fit(self: "A[int]") -> int:
        return self.x


a = A()
reveal_type(a)
a.fit()
reveal_type(a)
