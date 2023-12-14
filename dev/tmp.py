#!/usr/bin/env python

import pyarrow as pa
from pyarrow.lib import ArrowNotImplementedError

i8 = pa.int8()
i64 = pa.int64()
i32 = pa.int32()
f64 = pa.float64()
td64 = pa.duration("s")
b = pa.bool_()

duration_arr = pa.array([-3, 2, -1, 1], type=td64)
int_arr = pa.array([-3, 2, -1, 1], type=i64)
float_arr = pa.array([-3, 2, -1, 1], type=f64)

# td64 = pa.int64()
# duration_arr = pa.array([-3, 2, -1, 1], type=i64)

unary_ops = [
    (pa.compute.negate, duration_arr, td64),
    (pa.compute.negate_checked, duration_arr, td64),
    (pa.compute.abs, duration_arr, td64),
    (pa.compute.abs_checked, duration_arr, td64),
    (pa.compute.sign, duration_arr, i8),
    # tests
    (pa.compute.is_null, duration_arr, b),
    (pa.compute.is_valid, duration_arr, b),
    (pa.compute.is_finite, duration_arr, b),
    (pa.compute.is_inf, duration_arr, b),
    (pa.compute.is_nan, duration_arr, b),
    (pa.compute.true_unless_null, duration_arr, b),
    # aggregations
    (pa.compute.max, duration_arr, td64),
    (pa.compute.min, duration_arr, td64),
    (pa.compute.sum, duration_arr, td64),
    # (pa.compute.min_max, duration_arr, td64),  # returns tuple
    # (pa.compute.quantile, duration_arr, td64),  # float for int
    # (pa.compute.approximate_median, duration_arr, td64),  # float for int
    # cumulative aggregations
    (pa.compute.cumulative_sum, duration_arr, td64),
    (pa.compute.cumulative_sum_checked, duration_arr, td64),
    (pa.compute.cumulative_min, duration_arr, td64),
    (pa.compute.cumulative_max, duration_arr, td64),
]


for op, operand, dtype in unary_ops:
    try:
        result = op(operand)
    except ArrowNotImplementedError as e:
        print(f" [ ] {op.__name__:<24}({operand.type!s:<11}) -> {dtype}")
    else:
        assert result.type == dtype, f"{op}: got {result.type} expected {dtype}"
        print(f" [x] {op.__name__:<24}({operand.type!s:<11}) -> {dtype}")


binary_ops = [
    # arithmetic
    (pa.compute.add, duration_arr, duration_arr, td64),
    (pa.compute.add_checked, duration_arr, duration_arr, td64),
    (pa.compute.subtract, duration_arr, duration_arr, td64),
    (pa.compute.subtract_checked, duration_arr, duration_arr, td64),
    (pa.compute.multiply, duration_arr, int_arr, td64),
    (pa.compute.multiply_checked, duration_arr, int_arr, td64),
    (pa.compute.divide, duration_arr, duration_arr, f64),
    (pa.compute.divide, duration_arr, int_arr, td64),
    (pa.compute.divide_checked, duration_arr, duration_arr, f64),
    (pa.compute.divide_checked, duration_arr, int_arr, td64),
    # comparisons
    (pa.compute.less, duration_arr, duration_arr, b),
    (pa.compute.less_equal, duration_arr, duration_arr, b),
    (pa.compute.greater, duration_arr, duration_arr, b),
    (pa.compute.greater_equal, duration_arr, duration_arr, b),
    (pa.compute.equal, duration_arr, duration_arr, b),
    (pa.compute.not_equal, duration_arr, duration_arr, b),
    # min/max
    (pa.compute.max_element_wise, duration_arr, duration_arr, td64),
    (pa.compute.min_element_wise, duration_arr, duration_arr, td64),
    # containment
    (pa.compute.is_in, duration_arr, duration_arr, b),
    (pa.compute.index_in, duration_arr, duration_arr, i32),
]

for op, lhs, rhs, dtype in binary_ops:
    try:
        result = op(lhs, rhs)
    except ArrowNotImplementedError as e:
        print(f" [ ] {op.__name__:<20}({lhs.type!s:<11}, {rhs.type!s:<11}) -> {dtype}")
    else:
        assert result.type == dtype, f"{op}: got {result.type} expected {dtype}"
        print(f" [x] {op.__name__:<20}({lhs.type!s:<11}, {rhs.type!s:<11}) -> {dtype}")
