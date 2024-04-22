import pandas as pd

index = pd.DatetimeIndex(["2022-01-01T10:00:00", "2022-01-01T10:00:10"]).astype(
    "timestamp[s][pyarrow]"
)

assert isinstance(index, pd.DatetimeIndex)
# print(index.__doc__)
assert hasattr(index, "freq")
