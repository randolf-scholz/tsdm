# settings.py


from dataclasses import dataclass
from typing import ClassVar, Final


@dataclass
class Foo:
    x: Final[ClassVar[int]] = 42
    y: ClassVar[Final[int]] = 42
