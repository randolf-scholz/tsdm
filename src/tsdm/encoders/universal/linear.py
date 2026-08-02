r"""Linear data encoders."""

__all__ = [
    # Classes
    "LinearScaler",
    "MinMaxScaler",
    "StandardScaler",
]

from dataclasses import KW_ONLY, dataclass
from typing import Any, Self, cast, overload

from numerical_types import FloatArray
from tsdm.backend import Backend, get_backend
from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder
from tsdm.linalg.utils import invert_axis_selection, reduce_axes
from tsdm.types.aliases import Axis
from tsdm.utils.decorators import pprint_repr


@overload
def _reduce_param(param: float, selection: Any) -> float: ...
@overload
def _reduce_param[T: FloatArray](param: T, selection: Any) -> T: ...
def _reduce_param[T: FloatArray](param: float | T, selection: Any) -> float | T:
    r"""Perform a reduction on a parameter.

    For example, given tensor T, axis and selection, then this returns the slice of the tensor
    that satisfies the selection.
    """
    match param:
        case (int() | float()) as scalar:
            # NOTE: need to test int | float because of typing issues.
            # FIXME: https://github.com/python/typing/issues/1746
            return scalar
        case scalar if len(scalar.shape) == 0:
            return scalar
        case tensor:
            return tensor[selection]


@pprint_repr
@dataclass(init=False)
class LinearScaler[Arr: FloatArray](FittableEncoder[Arr, Arr]):
    r"""Maps the data linearly $x ↦ σ⋅x + μ$.

    Args:
        loc: the offset.
        scale: the scaling factor.
        axis: the axis along which to perform the operation. both μ and σ must have
            shapes that can be broadcasted to the shape of the data along these axes.
    """

    loc: Arr  # NDArray[np.number] | Tensor
    scale: Arr  # NDArray[np.number] | Tensor
    r"""The scaling factor."""

    axis: Axis
    r"""Over which axis to perform the scaling."""
    backend: Backend[Arr] = UNDEFINED
    r"""The backend of the encoder."""

    def __init__(
        self,
        loc: float | Arr = 0.0,
        scale: float | Arr = 1.0,
        *,
        axis: Axis = None,
    ) -> None:
        r"""Initialize the MinMaxScaler."""
        self.loc = cast("Arr", loc)
        self.scale = cast("Arr", scale)
        self.axis = axis

        if axis is not None:
            raise NotImplementedError("Axis not implemented yet.")

    def __getitem__(self, item: Any, /) -> Self:
        r"""Return a slice of the LinearScaler.

        Args:
            item: the slice, which is taken directly from the parameters.
                If the parameters are scalars, then we return the same scaler.
                E.g. taking slice encoder[:5] when loc and scale are scalars simply returns the same encoder.
                However, encoder[5] will have to modify the axis, since the new encoder will only operate
                on the 5th-entry along the first axis.

        Examples:
            - axis is (-2, -1) and data shape is (10, 20, 30, 40). Then
              loc/scale must be broadcastable to (30, 40), i.e. allowed shapes are
              (), (1,), (1,1), (30,), (30,1), (1,40), (30,40).
        """
        axis = reduce_axes(self.axis, item)
        loc = _reduce_param(self.loc, item)
        scale = _reduce_param(self.scale, item)

        # initialize the new encoder
        encoder = self.__class__(loc=loc, scale=scale, axis=axis)
        encoder.backend = self.backend
        return encoder

    def fit(self, data: Arr, /) -> None:
        self.backend: Backend[Arr] = get_backend(data)

    def encode(self, data: Arr, /) -> Arr:
        return data * self.scale + self.loc

    def decode(self, data: Arr, /) -> Arr:
        return (data - self.loc) / self.scale


@pprint_repr
@dataclass(init=False)
class StandardScaler[Arr: FloatArray](FittableEncoder[Arr, Arr]):
    r"""Transforms data linearly x ↦ (x-μ)/σ.

    axis: tuple[int, ...] determines the shape of the mean and stdv.
    """

    mean: float | Arr = 0.0
    r"""The mean value."""
    stdv: float | Arr = 1.0
    r"""The standard-deviation."""

    _: KW_ONLY

    axis: Axis = ()
    r"""The axis to perform the scaling. If None, automatically select the axis."""
    backend: Backend[Arr] = UNDEFINED
    r"""The backend of the encoder."""

    def __init__(
        self,
        mean: float | Arr = UNDEFINED,
        stdv: float | Arr = UNDEFINED,
        *,
        axis: Axis = (),
    ) -> None:
        self.mean = cast("Arr", mean)
        self.stdv = cast("Arr", stdv)
        self.axis = axis
        self.mean_learnable = mean is UNDEFINED
        self.stdv_learnable = stdv is UNDEFINED

    def __getitem__(self, item: Any, /) -> Self:
        r"""Return a slice of the Standardizer."""
        mean = _reduce_param(self.mean, item)
        stdv = _reduce_param(self.stdv, item)
        axis = reduce_axes(self.axis, item)

        # initialize the new encoder
        encoder = self.__class__(mean=mean, stdv=stdv, axis=axis)
        encoder.backend = self.backend
        return encoder

    def fit(self, data: Arr, /) -> None:
        # switch the backend
        self.backend: Backend[Arr] = get_backend(data)

        # universal fitting procedure
        axes = invert_axis_selection(self.axis, ndim=len(data.shape))

        if self.mean_learnable:
            self.mean = self.backend.nanmean(data, axis=axes)

        if self.stdv_learnable:
            self.stdv = self.backend.nanstd(data, axis=axes)

    def encode(self, data: Arr, /) -> Arr:
        # TODO: consider adding broadcasting
        #   1. broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        #   2. return (data - self.mean[broadcast]) / self.stdv[broadcast]
        return (data - self.mean) / self.stdv

    def decode(self, data: Arr, /) -> Arr:
        # TODO: consider adding broadcasting
        #   1. broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        #   2. return data * self.stdv[broadcast] + self.mean[broadcast]
        return data * self.stdv + self.mean


@pprint_repr
@dataclass(init=False)
class MinMaxScaler[Arr: FloatArray](FittableEncoder[Arr, Arr]):
    r"""Linearly transforms [x_min, x_max] to [y_min, y_max] (default: [0, 1]).

    If x_min and/or x_max are provided at initialization, they are marked as
    "fitted" and will not be re-computed during the fit method.

    Examples:
        - axis=() (default): fit one scaler for all elements.
        - axis=-1: fit one scaler per channel.
        - axis=(-2, -1): fit one scaler per image.
        - axis=None: fit one scaler per element.

    Note:
        In the edge case when fitting to a single value, we set the scale such that
        the encoded value is ½(y_min + y_max).

    Note:
        Generally, the transformation is given by the formula:

        .. math:: x ↦ \frac{x - xₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(yₘₐₓ - yₘᵢₙ) + yₘᵢₙ

        We transform the formula, to cater to the edge case, by extending with the
        average of the min and max x-values:

        .. math::
            \frac{yₘₐₓ - yₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(x - xₘᵢₙ) + yₘᵢₙ \\
            = \frac{x +½(xₘₐₓ - xₘᵢₙ) -½(xₘₐₓ - xₘᵢₙ) - xₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(yₘₐₓ - yₘᵢₙ) + yₘᵢₙ \\
            = ½(yₘₐₓ - yₘᵢₙ) + \frac{x - ½(xₘₐₓ + xₘᵢₙ)}{xₘₐₓ - xₘᵢₙ}(yₘₐₓ - yₘᵢₙ) + yₘᵢₙ \\
            = \frac{yₘₐₓ - yₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(x - x̄) + ȳ \\
            = γ⋅(x - x̄) + ȳ

        In particular, when xₘᵢₙ = xₘₐₓ, we have x̄ = xₘᵢₙ = xₘₐₓ, and formally set
        the scale to 1, making the transform x ↦ x + (ȳ - x̄)

    Note:
        Whatever formula we use, it should be ensured that when x∈[x_min, x_max],
        then y∈[y_min, y_max], i.e. the result should be within bounds.
        This might be violated due to numerical roundoff, so we need to be careful.
    """

    ymin: Arr = UNDEFINED  # or ScalarType.
    ymax: Arr = UNDEFINED  # or ScalarType.
    xmin: Arr = UNDEFINED  # or ScalarType.
    xmax: Arr = UNDEFINED  # or ScalarType.

    axis: Axis = ()
    r"""Over which axis to perform the scaling."""
    safe_computation: bool = True
    r"""Whether to ensure that the bounds are not violated due to roundoff."""
    backend: Backend[Arr] = UNDEFINED
    r"""The backend of the encoder."""

    def __init__(
        self,
        ymin: float | Arr = 0.0,
        ymax: float | Arr = 1.0,
        *,
        xmin: None | float | Arr = None,
        xmax: None | float | Arr = None,
        axis: Axis = (),
        safe_computation: bool = True,
    ) -> None:
        self.axis = axis
        self.safe_computation = safe_computation
        self.ymax = cast("Arr", ymax)
        self.ymin = cast("Arr", ymin)

        self.xmin_learnable = xmin is None
        self.xmax_learnable = xmax is None
        self.xmin = UNDEFINED if xmin is None else cast("Arr", xmin)
        self.xmax = UNDEFINED if xmax is None else cast("Arr", xmax)
        self.xbar: Arr = UNDEFINED  # or ScalarType.
        self.ybar: Arr = UNDEFINED  # or ScalarType.
        self.scale: Arr = UNDEFINED  # or ScalarType.

        # set derived parameters
        self.set_derived_fields()

        # set initial backend
        self.switch_backend(get_backend(self.params))

    def set_derived_fields(self) -> None:
        if self.xmin is UNDEFINED or self.xmax is UNDEFINED:
            self.xbar = UNDEFINED
            self.ybar = UNDEFINED
            self.scale = UNDEFINED
        else:
            # compute the midpoints
            self.xbar = (self.xmax + self.xmin) / 2
            self.ybar = (self.ymax + self.ymin) / 2
            # compute the scale
            self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def __getitem__(self, item: Any, /) -> Self:
        r"""Return a slice of the MinMaxScaler."""
        xmin = _reduce_param(self.xmin, item)
        xmax = _reduce_param(self.xmax, item)
        ymin = _reduce_param(self.ymin, item)
        ymax = _reduce_param(self.ymax, item)
        axis = reduce_axes(self.axis, item)

        # initialize the new encoder
        encoder = self.__class__(ymin, ymax, xmin=xmin, xmax=xmax, axis=axis)
        encoder.backend = self.backend
        return encoder

    def fit(self, data: Arr, /) -> None:
        # switch the backend
        self.backend: Backend[Arr] = get_backend(data)

        # skip if the parameters are not learnable
        if not (self.xmin_learnable or self.xmax_learnable):
            return

        # invert axes selection
        axes = invert_axis_selection(self.axis, ndim=len(data.shape))
        if self.xmin_learnable:
            self.xmin = self.backend.nanmin(data, axis=axes)
        if self.xmax_learnable:
            self.xmax = self.backend.nanmax(data, axis=axes)

        self.ymin = self.backend.to_tensor(self.ymin)
        self.ymax = self.backend.to_tensor(self.ymax)
        self.xmin = self.backend.to_tensor(self.xmin)
        self.xmax = self.backend.to_tensor(self.xmax)

        # broadcast y to the same shape as x
        self.ymin = self.ymin + 0.0 * self.xmin
        self.ymax = self.ymax + 0.0 * self.xmax
        self.set_derived_fields()

        dx = self.xmax - self.xmin
        dy = self.ymax - self.ymin
        scale = dy / dx
        self.scale = self.backend.where(dx != 0, scale, scale**0)
        self.recast_parameters()

    def encode(self, x: Arr, /) -> Arr:
        r"""Maps [xₘᵢₙ, xₘₐₓ] to [yₘᵢₙ, yₘₐₓ]."""
        y = (x - self.xbar) * self.scale + self.ybar
        if self.safe_computation:
            return self.project_encoding(x, y)
        return y

    def project_encoding(self, x: Arr, y: Arr, /) -> Arr:
        r"""Ensures that the encoded values are within the bounds.

        .. math::
            x < xₘᵢₙ &⟹ y < yₘᵢₙ  \\
            x > xₘₐₓ &⟹ y > yₘₐₓ  \\
            x∈[xₘᵢₙ, xₘₐₓ] &⟹ y∈[yₘᵢₙ, yₘₐₓ]
        """
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        backend = self.backend
        y = backend.where(x < xmin, backend.clip(y, None, ymin), y)
        y = backend.where(x > xmax, backend.clip(y, ymax, None), y)
        y = backend.where((x >= xmin) & (x <= xmax), backend.clip(y, ymin, ymax), y)
        return y

    def decode(self, y: Arr, /) -> Arr:
        r"""Maps [yₘᵢₙ, yₘₐₓ] to [xₘᵢₙ, xₘₐₓ]."""
        x = (y - self.ybar) / self.scale + self.xbar
        if self.safe_computation:
            return self.project_decoding(y, x)
        return x

    def project_decoding(self, y: Arr, x: Arr, /) -> Arr:
        r"""Projects the decoded values to the bounds.

        .. math::
            y < yₘᵢₙ &⟹ x < xₘᵢₙ  \\
            y > yₘₐₓ &⟹ x > xₘₐₓ  \\
            y∈[yₘᵢₙ, yₘₐₓ] &⟹ x∈[xₘᵢₙ, xₘₐₓ]
        """
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        backend = self.backend
        x = backend.where(y < ymin, backend.clip(x, None, xmin), x)
        x = backend.where(y > ymax, backend.clip(x, xmax, None), x)
        x = backend.where((y >= ymin) & (y <= ymax), backend.clip(x, xmin, xmax), x)
        return x

    # region parameters ----------------------------------------------------------------

    def recompute_params(self) -> None:
        r"""Computes derived parameters from the base parameters."""
        self.xbar = (self.xmax + self.xmin) / 2
        self.ybar = (self.ymax + self.ymin) / 2
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def switch_backend(self, backend: str | Backend) -> None:
        r"""Switch the backend of the scaler."""
        self.backend: Backend[Arr] = Backend(backend)
        self.recast_parameters()

    def recast_parameters(self) -> None:
        r"""Recast the parameters to the current backend."""
        # switch the backend of the parameters
        self.xmin = self.backend.to_tensor(self.xmin)
        self.xmax = self.backend.to_tensor(self.xmax)
        self.ymin = self.backend.to_tensor(self.ymin)
        self.ymax = self.backend.to_tensor(self.ymax)
        self.xbar = self.backend.to_tensor(self.xbar)
        self.ybar = self.backend.to_tensor(self.ybar)
        self.scale = self.backend.to_tensor(self.scale)

    # endregion parameters -------------------------------------------------------------
