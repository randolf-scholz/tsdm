r"""Linear Algebra Subroutines."""

__all__ = [
    # submodules
    "collate",
    "logical_operators",
    "matrix_functions",
    "tensor_functions",
    "utils",
]

from . import collate, logical_operators, matrix_functions, tensor_functions, utils
from .collate import *  # ruff: ignore[F403]
from .logical_operators import *  # ruff: ignore[F403]
from .matrix_functions import *  # ruff: ignore[F403]
from .tensor_functions import *  # ruff: ignore[F403]
from .utils import *  # ruff: ignore[F403]

__all__ += collate.__all__
__all__ += logical_operators.__all__
__all__ += matrix_functions.__all__
__all__ += tensor_functions.__all__
__all__ += utils.__all__
