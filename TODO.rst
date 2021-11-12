ToDo's
======

- StandardScalar is not properly vectorized. Do custom implementation instead.
- Exact loss function? Forecast?

- Without Standard-scaling
- RMSE **NOT** channel wise, but flattened
- accumulate cross-validation: RMSE
- horizon 1h / 2h / 6h  ↭ avg number of timesteps
- horizons: 150+100, 300+300, 400+600
- In TS split: first snippet that fits until last snippet


- Time Encoding?

- negative values: truncate to zero!

timeseries.min()

    Flow_Air                           0.000000
    StirringSpeed                      0.000000
    Temperature                       32.689999
    Acetate                           -0.257518
    Base                               0.000000
    Cumulated_feed_volume_glucose      3.000000
    Cumulated_feed_volume_medium       5.882958
    DOT                                0.000000
    Glucose                           -0.094553
    OD600                             -0.962500
    Probe_Volume                     200.000000
    pH                                 0.000000
    Fluo_GFP                        -250.000000
    InducerConcentration               0.000000
    Volume                            -2.235569


Sensor spot
-



Current WorkFLow

KIWI data - cut off negative values

- push tsdm with KIWI task
- run linodenet on KIWI data
- create graphs
- encode metadata
- results with metadata


- results on Electricity, ETTh, Traffic


ADD
ECL : https://drive.google.com/file/d/1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP/view?usp=sharing
Weather: https://drive.google.com/file/d/1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG/view?usp=sharing