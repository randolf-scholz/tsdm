#!/usr/bin/env python

from pandas import DataFrame, Index, MultiIndex, Series
from typing_extensions import assert_type, reveal_type

# Getting values
df = DataFrame(
    [[1, 2], [4, 5], [7, 8]],
    index=["cobra", "viper", "sidewinder"],
    columns=["max_speed", "shield"],
)

assert_type(df, DataFrame)
assert_type(df.loc["viper"], Series)
assert_type(df.loc[["viper", "sidewinder"]], DataFrame)
assert_type(df.loc["cobra", "shield"], int)
assert_type(df.loc["cobra":"viper", "max_speed"], Series)
assert_type(df.loc[[False, False, True]], DataFrame)
assert_type(
    df.loc[Series([False, True, False], index=["viper", "sidewinder", "cobra"])],
    DataFrame,
)
assert_type(df.loc[Index(["cobra", "viper"], name="foo")], DataFrame)
assert_type(df.loc[df["shield"] > 6], DataFrame)
assert_type(df.loc[df["shield"] > 6, ["max_speed"]], Series)
assert_type(df.loc[lambda frame: frame["shield"] == 8], DataFrame)

# Setting values
df.loc[["viper", "sidewinder"], ["shield"]] = 50
df.loc["cobra"] = 10
df.loc[:, "max_speed"] = 30
df.loc[df["shield"] > 35] = 0

# Getting values on a DataFrame with an index that has integer labels
df = DataFrame(
    [[1, 2], [4, 5], [7, 8]], index=[7, 8, 9], columns=["max_speed", "shield"]
)
assert_type(df, DataFrame)
assert_type(df.loc[7:9], DataFrame)
μ̄

# Getting values with a MultiIndex
tuples = [
    ("cobra", "mark i"),
    ("cobra", "mark ii"),
    ("sidewinder", "mark i"),
    ("sidewinder", "mark ii"),
    ("viper", "mark ii"),
    ("viper", "mark iii"),
]
index = MultiIndex.from_tuples(tuples)
values = [[12, 2], [0, 4], [10, 20], [1, 4], [7, 1], [16, 36]]
df = DataFrame(values, columns=["max_speed", "shield"], index=index)
assert_type(df, DataFrame)
assert_type(df.loc["cobra"], DataFrame)
assert_type(df.loc[("cobra", "mark ii")], Series)
assert_type(df.loc["cobra", "mark i"], Series)
assert_type(df.loc[[("cobra", "mark ii")]], DataFrame)
assert_type(df.loc[[("cobra", "mark ii"), "shield"]], int)
assert_type(df.loc[("cobra", "mark i"):"viper"], DataFrame)
assert_type(df.loc[("cobra", "mark i"):("viper", "mark ii")], DataFrame)


# Extra example
# index = MultiIndex.from_tuples([(0, 0), (0, 1)])
# df = DataFrame(range(2), index=index)
# key = (0, 0)
# x = df.loc[key].copy()  # raises [union-attr] Item has no attribute "copy"
