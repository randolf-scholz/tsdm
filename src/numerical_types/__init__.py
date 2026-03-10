r"""Type definitions for scalars, arrays, and series."""

__all__ = [
    # submodules
    "arrays",
    "mixins",
    "scalars",
    "series",
    "tables",
]

from numerical_types import arrays, mixins, scalars, series, tables
from numerical_types.arrays import *  # noqa: F403
from numerical_types.mixins import *  # noqa: F403
from numerical_types.scalars import *  # noqa: F403
from numerical_types.series import *  # noqa: F403
from numerical_types.tables import *  # noqa: F403

__all__ += arrays.__all__
__all__ += mixins.__all__
__all__ += scalars.__all__
__all__ += series.__all__
__all__ += tables.__all__
