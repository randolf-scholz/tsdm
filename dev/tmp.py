import pandas as pd

DATA = r"""
time, value
2022-01-01T10:00:00, 10
2022-01-01T10:00:30, 15
2022-01-01T10:01:00, 14
2022-01-01T10:01:30, 20
"""


pd.read_csv(DATA)
