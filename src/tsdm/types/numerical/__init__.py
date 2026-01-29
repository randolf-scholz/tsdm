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


from tsdm.types.numerical.arrays import (
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
from tsdm.types.numerical.scalars import (
    BoolScalar,
    ComplexScalar,
    FloatScalar,
    IntScalar,
    ScalarType,
    SpanLikeScalar,
    TimeLikeScalar,
)
from tsdm.types.numerical.series import (
    BooleanSeries,
    ComplexSeries,
    DatetimeSeries,
    FloatSeries,
    IntegerSeries,
    SeriesType,
    TimedeltaSeries,
)
from tsdm.types.numerical.tables import TableType
