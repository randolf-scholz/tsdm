r"""Implementations for torch backend."""

__all__ = [
    # Constants
    "EPS",
    # Functions
    "apply_along_axes",
    "copy_like",
    "drop_null",
    "nanmax",
    "nanmin",
    "nanstd",
    "scalar",
    # utils
    "initialize_from_config",
    "autojit",
    "lazy_jit_torch",
]

from collections.abc import Callable as Fn
from functools import wraps
from importlib import import_module
from typing import Any, Final, Self

import torch
from numpy.typing import ArrayLike
from torch import Tensor, jit, nn

from tsdm.config import CONFIG
from tsdm.types.aliases import Axis

EPS: Final[dict[torch.dtype, float]] = {
    torch.bfloat16   : 1e-2,
    torch.complex128 : 1e-15,
    torch.complex32  : 1e-3,
    torch.complex64  : 1e-6,
    torch.float16    : 1e-3,
    torch.float32    : 1e-6,
    torch.float64    : 1e-15,
}  # fmt: skip
r"""CONST: Default epsilon for each dtype."""


def scalar(x: Any, /, dtype: Any) -> Any:
    return torch.tensor(x, dtype=dtype).item()


def drop_null(x: Tensor, /) -> Tensor:
    r"""Drop `NaN` values from a tensor, flattening it."""
    return x[~torch.isnan(x)].flatten()


def nanmin(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanmin`."""
    return torch.amin(
        torch.where(torch.isnan(x), float("+inf"), x),
        dim=axis,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        keepdim=keepdims,
    )


def nanmax(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanmax`."""
    return torch.amax(
        torch.where(torch.isnan(x), float("-inf"), x),
        dim=axis,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        keepdim=keepdims,
    )


def nanstd(x: Tensor, /, *, axis: Axis = None, keepdims: bool = False) -> Tensor:
    r"""Analogue to `numpy.nanstd`."""
    r = x - torch.nanmean(x, dim=axis, keepdim=True)
    return torch.sqrt(
        torch.nanmean(
            r.pow(2),
            dim=axis,
            keepdim=keepdims,
        )
    )


def copy_like(x: ArrayLike, ref: Tensor, /) -> Tensor:
    r"""Return a tensor of the same dtype and other options as `ref`."""
    return torch.tensor(x, dtype=ref.dtype, device=ref.device)


def apply_along_axes(op: Fn[..., Tensor], /, *tensors: Tensor, axis: Axis) -> Tensor:
    r"""Apply a function to multiple tensors along axes.

    Assumptions:
    - All tensors must have the same shape.
    - The operator `op` acts on the last `len(axis)` axes of the tensors.
    - The operator `op` does not change the shape of the tensors.
    """
    if len(tensors) < 1:
        raise ValueError("At least one tensor is required!")
    if len({t.shape for t in tensors}) != 1:
        raise ValueError("All tensors must have the same shape!")

    # we move the target axes to the front, apply the operation,
    # then move them back to their original position
    rank = len(tensors[0].shape)
    target_axes = (
        () if axis is None
        else (axis % rank,) if isinstance(axis, int)
        else tuple(ax % rank for ax in axis)
    )  # fmt: skip
    other_axes = tuple(ax for ax in range(rank) if ax not in target_axes)
    source = tuple(range(rank))
    inv = target_axes + other_axes  # inverse permutation
    dest = tuple(sorted(source, key=inv.__getitem__))
    tensors = tuple(torch.moveaxis(tensor, source, dest) for tensor in tensors)
    result = op(*tensors)
    result = torch.moveaxis(result, source, dest)
    return result


def lazy_jit_torch[**P, R](func: Fn[P, R], /) -> Fn[P, R]:  # +R
    r"""Create decorator to lazily compile a function with `torch.jit.script`."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # script the original function if it hasn't been scripted yet
        if wrapper.scripted is None:  # type: ignore[attr-defined]
            wrapper.scripted = jit.script(wrapper.original_fn)  # type: ignore[attr-defined]
        return wrapper.scripted(*args, **kwargs)  # type: ignore[attr-defined]

    wrapper.original_fn = func  # type: ignore[attr-defined]
    wrapper.scripted = None  # type: ignore[attr-defined]
    wrapper.script_if_tracing_wrapper = True  # type: ignore[attr-defined]
    return wrapper


def autojit[M: nn.Module](base_class: type[M], /) -> type[M]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule: ...


        model = jit.script(MyModule())

    and

    .. code-block:: python

        @autojit
        class MyModule: ...


        model = MyModule()

    are (roughly?) equivalent
    """
    if not isinstance(base_class, type):
        raise TypeError("Expected a class.")
    if not issubclass(base_class, nn.Module):
        raise TypeError("Expected a subclass of nn.Module.")

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type,misc]
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            # Note: If __new__() does not return an instance of cls,
            #   then the new instance's __init__() method will not be invoked.
            instance = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted = jit.script(instance)
                return scripted  # type: ignore[return-value]
            return instance  # type: ignore[return-value]

    if not isinstance(WrappedClass, type):
        raise TypeError(f"Expected a class, got {WrappedClass}.")
    if not issubclass(WrappedClass, base_class):
        raise TypeError(f"Expected {WrappedClass} to be a subclass of {base_class}.")

    return WrappedClass  # pyright: ignore[reportReturnType]


def initialize_from_config(config: dict[str, Any], /) -> nn.Module:
    r"""Initialize `nn.Module` from a config object."""
    conf = config.copy()
    cls_name: str = conf.pop("__name__")
    module_name: str = conf.pop("__module__")

    # drop other dunder keys
    opts = {k: v for k, v in conf.items() if not k.startswith("__")}

    # import module and class
    module = import_module(module_name)
    cls = getattr(module, cls_name)

    # initialize class with options
    try:
        obj = cls(**opts)
    except Exception as exc:
        exc.add_note(f"Failed to initialize {cls_name} with {opts}.")
        raise

    return obj
