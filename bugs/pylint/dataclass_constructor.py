#!/usr/bin/env python

import logging
from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, ClassVar, dataclass
from functools import wraps
from typing import Any, Generic, Mapping, ParamSpec, Self, Sequence

P = ParamSpec("P")


class CallbackMetaclass(ABCMeta):
    """Metaclass for callbacks."""

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")


@dataclass(repr=False)
class Foo(Generic[P], metaclass=CallbackMetaclass):
    LOGGER: ClassVar[logging.Logger]
    """The debug-logger for the callback."""
    required_kwargs: ClassVar[set[str]]
    """The required kwargs for the callback."""
    _: KW_ONLY

    freq: int = 0

    def __init_subclass__(cls) -> None:
        """Automatically set the required kwargs for the callback."""
        cls.required_kwargs = {"i"}

        @wraps(cls.callback)
        def __call__(self: Self, i: int, /, **kwargs: P.kwargs) -> None:
            """Log something at the end of a batch/epoch."""
            if i % self.frequency == 0:
                self.callback(i, **kwargs)
            else:
                self.LOGGER.debug("Skipping callback.")

        cls.__call__ = __call__  # type: ignore[method-assign]
        super().__init_subclass__()

    def __call__(self, i: int, /, **kwargs: P.kwargs) -> None:
        """Log something at the end of a batch/epoch."""

    @abstractmethod
    def callback(self, i: int, /, **kwargs: P.kwargs) -> None:
        """Log something at the end of a batch/epoch."""


@dataclass
class Bar(Foo):
    metrics: Sequence[str | type | type[type]] | Mapping[str, str | type | type[type]]
    writer: Any

    _: KW_ONLY

    key: str = ""

    def callback(self, i: int, /, **kwargs: Any) -> None:
        pass


obj = Bar(3, 4)
obj.LOGGER.info("Hello, world!")


if __name__ == "__main__":
    pass
