#!/usr/bin/env python

import pandas as pd

anomalies = pd.DatetimeIndex({
    "Jan. 1, 2008": "New Year’s Day",
    "Jan. 21, 2008": "Martin Luther King Jr. Day",
    "Feb. 18, 2008": "Washington’s Birthday",
    "Mar. 9, 2008": "Anomaly day",
    "May 26, 2008": "Memorial Day",
    "Jul. 4, 2008": "Independence Day",
    "Sep. 1, 2008": "Labor Day",
    "Oct. 13, 2008": "Columbus Day",
    "Nov. 11, 2008": "Veterans Day",
    "Nov. 27, 2008": "Thanksgiving",
    "Dec. 25, 2008": "Christmas Day",
    "Jan. 1, 2009": "New Year’s Day",
    "Jan. 19, 2009": "Martin Luther King Jr. Day",
    "Feb. 16, 2009": "Washington’s Birthday",
    "Mar. 8, 2009": "Anomaly day",
})

print(anomalies)
