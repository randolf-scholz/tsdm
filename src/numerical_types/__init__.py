r"""Type definitions for scalars, arrays, and series."""

__all__ = [
    # submodules
    "arrays",
    "mixins",
    "scalars",
    "series",
    "tables",
]

from . import arrays, mixins, scalars, series, tables
from .arrays import *  # ruff: ignore[F403]
from .mixins import *  # ruff: ignore[F403]
from .scalars import *  # ruff: ignore[F403]
from .series import *  # ruff: ignore[F403]
from .tables import *  # ruff: ignore[F403]

__all__ += arrays.__all__
__all__ += mixins.__all__
__all__ += scalars.__all__
__all__ += series.__all__
__all__ += tables.__all__
