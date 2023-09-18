#!/usr/bin/env python3

from types import EllipsisType

import numpy as np
import torch
from numpy._typing import _NestedSequence

#
# x = ...
# match x:
#     case EllipsisType():
#         print(0)
# import numpy as np
#


x: _NestedSequence = torch.tensor([1, 2, 3])
y: _NestedSequence = np.array([1, 2, 3])
z: _NestedSequence = [[[[1]]]]

# reveal_type(np.ndarray.__getitem__)
#
#
# def __getitem__(
#     self,
#     key: (
#         NDArray[integer[Any]]
#         | NDArray[bool_]
#         | tuple[NDArray[integer[Any]] | NDArray[bool_], ...]
#     ),
# ) -> ndarray[Any, _DType_co]: ...
# @overload
# def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> Any: ...
# @overload
# def __getitem__(
#     self,
# ) -> ndarray[Any, _DType_co]: ...
# @overload
# def __getitem__(self: NDArray[void], key: str) -> NDArray[Any]: ...
# @overload
# def __getitem__(
#     self: NDArray[void], key: list[str]
# ) -> ndarray[_ShapeType, _dtype[void]]: ...
