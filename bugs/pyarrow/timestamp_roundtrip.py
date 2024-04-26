#!/usr/bin/env python

from datetime import datetime

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import parquet

dates = pa.array(
    [
        datetime(2021, 1, 1, 0, 0, 3),
        datetime(2021, 1, 1, 0, 0, 4),
        datetime(2021, 1, 1, 0, 0, 5),
    ],
    type=pa.timestamp("s"),
)

diff = pc.subtract(dates, dates)

table = pa.table({"time": diff})
print(table.schema)  # timestamp[s]
parquet.write_table(table, "timestamp_roundtrip.parquet")
table2 = parquet.read_table("timestamp_roundtrip.parquet")
print(table2.schema)  # timestamp[ms]
