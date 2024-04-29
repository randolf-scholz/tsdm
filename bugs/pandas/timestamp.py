import numpy as np
from pandas import Timedelta, Timestamp

t: Timestamp = Timestamp("2024-04-29T18:00:00")
t2: Timestamp = Timestamp(np.datetime64("nat"))
dt: Timedelta = Timedelta(1, "h")
dt2: Timedelta = Timedelta(np.timedelta64("nat"))
