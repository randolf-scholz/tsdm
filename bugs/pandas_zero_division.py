from pandas import Series, Timedelta, show_versions

unit = Timedelta("1s")
time = Series([Timedelta("0 days 00:00:00"), Timedelta("-1 days +23:59:59")])
time_arrow = time.astype("duration[us][pyarrow]")

print(time / unit)  # [0, -1]
print(time_arrow / unit)  # [0, NaN]
show_versions()
