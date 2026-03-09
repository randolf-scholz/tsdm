r"""Type definitions for scalars, arrays, and series."""

__all__ = [
    # scalars
    "ScalarType",
    "BoolScalar",
    "ComplexScalar",
    "FloatScalar",
    "IntScalar",
    "SpanLikeScalar",
    "TimeLikeScalar",
    "TimedeltaScalar",
    "DatetimeScalar",
    # arrays
    "ArrayType",
    "TimeLikeArray",
    "SpanLikeArray",
    "BooleanArray",
    "IntegerArray",
    "FloatArray",
    "ComplexArray",
    "TimedeltaArray",
    "DatetimeArray",
    # series
    "SeriesType",
    "BooleanSeries",
    "IntegerSeries",
    "FloatSeries",
    "ComplexSeries",
    "TimedeltaSeries",
    "DatetimeSeries",
    # tables
    "TableType",
]


from tsdm.experimental.types.arrays import (
    ArrayType,
    BooleanArray,
    ComplexArray,
    DatetimeArray,
    FloatArray,
    IntegerArray,
    SpanLikeArray,
    TimedeltaArray,
    TimeLikeArray,
)
from tsdm.experimental.types.scalars import (
    BoolScalar,
    ComplexScalar,
    DatetimeScalar,
    FloatScalar,
    IntScalar,
    ScalarType,
    SpanLikeScalar,
    TimedeltaScalar,
    TimeLikeScalar,
)
from tsdm.experimental.types.series import (
    BooleanSeries,
    ComplexSeries,
    DatetimeSeries,
    FloatSeries,
    IntegerSeries,
    SeriesType,
    TimedeltaSeries,
)
from tsdm.experimental.types.tables import TableType
