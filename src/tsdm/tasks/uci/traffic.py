r"""Tasks associated with the Traffic dataset."""

__all__ = [
    "Traffic_Lim2021",
    "Traffic_Yu2016",
]


class Traffic_Lim2021:
    r"""Experiments as performed by the "TFT" paper.

    Paper
    -----
    - | Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
      | Lim et al.
      | International Journal of Forecasting 2021
      | https://www.sciencedirect.com/science/article/pii/S0169207021000637

    Evaluation Protocol
    -------------------

    .. epigraph::

        Traffic: Tests on the Traffic dataset are also kept consistent with previous work, using
        500k training samples taken before 2008-06-15 as per [9], and key in the same way as the
        Electricity dataset. For testing, we use the 7 days immediately following the training set,
        and z-score normalization was applied across all entities. For inputs, we also take traffic
        occupancy, day-of-week, hour-of-day and a time index as real-valued inputs, and the
        entity identifier as a categorical variable.

    Test-Metric
    -----------
    ?

    Results
    -------
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | Model | ARIMA | ConvTrans | DSSM  | DeepAR | ETS   | MQRNN | Seq2Seq | TFT   | TRMF  |
    +=======+=======+===========+=======+========+=======+=======+=========+=======+=======+
    | P50   | 0.223 | 0.122     | 0.167 | 0.161  | 0.236 | 0.117 | 0.105   | 0.095 | 0.186 |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | P90   | 0.137 | 0.081     | 0.113 | 0.099  | 0.148 | 0.082 | 0.075   | 0.070 | NaN   |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    """


class Traffic_Yu2016:
    r"""Experiments as performed by the "TRMF" paper.

    Paper
    -----
    - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      | Hsiang-Fu Yu, Nikhil Rao, Inderjit S Dhillon
      | Advances in Neural Information Processing Systems 29 (NeurIPS 2016)
      | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html

    Evaluation Protocol
    -------------------

    .. epigraph::

        5.1 Forecasting
        [...]
        For electricity and traffic, we consider the 24-hour ahead forecasting task and use the last
        seven days as the test periods.

        A.1 Datasets and Evaluation Criteria
        [...]
        traffic 4 : A collection of 15 months of daily data from the California Department of
        Transportation. The data describes the occupancy rate, between 0 and 1, of different car
        lanes of San Francisco Bay Area freeways. The data was sampled every 10 minutes, and we
        again aggregate the columns to obtain hourly traffic data to finally get n = 963,
        T = 10, 560. The coefficient of variation for traffic is 0.8565.


    Test-Metric
    -----------

    **Normalized deviation (ND)**

    .. math::
        𝖭𝖣(Y, Ŷ) = \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t)∈Ω_\text{test}}|Ŷ_{it}-Y_{it}|\Big)
        \Big/ \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Y_{it}|\Big)

    **Normalized RMSE (NRMSE)**

    .. math::
        𝖭𝖱𝖬𝖲𝖤(Y, Ŷ) = \sqrt{\frac{1}{|Ω_\text{test}|}∑_{(i,t) ∈ Ω_\text{test}}|Ŷ_{it}-Y_{it}|^2}
        \Big/ \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i, t) ∈ Ω_\text{test}}|Y_{it}|\Big)

    Results
    -------
    +-------+-------+-------------+-------------+---------------+
    | Model | TRMF  | N-BEATS (G) | N-BEATS (I) | N-BEATS (I+G) |
    +=======+=======+=============+=============+===============+
    | ND    | 0.187 | 0.112       | 0.110       | 0.111         |
    +-------+-------+-------------+-------------+---------------+
    """
