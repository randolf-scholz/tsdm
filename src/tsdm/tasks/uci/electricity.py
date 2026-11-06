r"""Tasks associated with the Electricity dataset."""

__all__ = [
    # CLASSES
    "Electricity_Sundar2018",
    "Electricity_Salinas2020",
    "Electricity_Yu2016",
    "Electricity_Li2019",
]


class Electricity_Sundar2018:
    r"""Experiments as performed by the "DeepState" paper.

    References:
        - Deep State Space Models for Time Series Forecasting <br>
          Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, im Januschowski <br>
          Advances in Neural Information Processing Systems 31 (NeurIPS 2018) <br>
          <https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html>

    ### Evaluation Protocol

    > We train each method on all time series of these dataset but vary the size of the training
    > range $Tᵢ∈\{14, 21, 28\}$ days. We evaluate all the methods on the next $τ=7$ days after the
    > forecast start time using the standard p50 and p90- quantile losses.

    ### Horizons and Split

    The observation horizons are 14, 21, or 28 days; the forecasting horizon is
    7 days. The split date remains unverified: the existing notes cite 2014-09-01
    from the N-BEATS comparison, where it appears in example plots in Appendix A.5
    (Figures 4 and 5).
    """


class Electricity_Salinas2020:
    r"""Experiments as performed by the "DeepAR" paper.

    References:
        - DeepAR: Probabilistic forecasting with autoregressive recurrent networks <br>
          David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski <br>
          International Journal of Forecasting 36 (2020) <br>
          <https://www.sciencedirect.com/science/article/pii/S0169207019301888>

    ### Evaluation Protocol

    > For electricity, we train with data between 2014-01-01 and 2014-09-01, for traffic we train
    > all the data available before 2008-06-15. The results for electricity and traffic are
    > computed using rolling window predictions done after the last point seen in training as
    > described in [23]. We do not retrain our model for each window, but use a single model
    > trained on the data before the first prediction window.
    """


class Electricity_Yu2016:
    r"""Experiments as performed by the "TRMF" paper.

    References:
        - Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <br>
          Hsiang-Fu Yu, Nikhil Rao, Inderjit S Dhillon <br>
          Advances in Neural Information Processing Systems 29 (NeurIPS 2016) <br>
          https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html

    ### Evaluation Protocol

    > 5.1 Forecasting
    > [...]
    > For electricity and traffic, we consider the 24-hour ahead forecasting task and use last
    > seven days as the test periods.
    >
    > A.1 Datasets and Evaluation Criteria
    > [...]
    > electricity 3 : the electricity usage in kW recorded every 15 minutes, for n = 370 clients.
    > We convert the data to reflect hourly consumption, by aggregating blocks of 4 columns,
    > to obtain T = 26, 304. The coefficient of variation for electricity is 6.0341.

    ### Test Metrics

    Let $Ω$ be the non-empty set of test observations and $n = |Ω|$.
    Both metrics require $∑_{(i,t) ∈ Ω} |Yᵢₜ| > 0$.

    **Normalized deviation (ND)**

    $$ 𝖭𝖣(Y, Ŷ) = \frac{∑_{(i,t) ∈ Ω} |Ŷᵢₜ - Yᵢₜ|}{∑_{(i,t) ∈ Ω} |Yᵢₜ|} $$

    **Normalized RMSE (NRMSE)**

    $$ 𝖭𝖱𝖬𝖲𝖤(Y, Ŷ) = \frac{\sqrt{n ∑_{(i,t) ∈ Ω} |Ŷᵢₜ - Yᵢₜ|²}}{∑_{(i,t) ∈ Ω} |Yᵢₜ|} $$

    ### Results

    | Model | TRMF  | N-BEATS (G) | N-BEATS (I) | N-BEATS (I+G) |
    |-------|-------|-------------|-------------|---------------|
    | ND    | 0.255 | 0.171       | 0.185       | 0.111         |
    """


class Electricity_Li2019:
    r"""Experiments as performed by the "LogSparseTransformer" paper.

    References:
        - Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting <br>
          Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, Xifeng Yan <br>
          Advances in Neural Information Processing Systems 32 (NeurIPS 2019) <br>
          https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html


    ### Evaluation Protocol

    > For short-term forecasting, we evaluate rolling-day forecasts for seven days ( i.e.,
    > prediction horizon is one day and forecasts start time is shifted by one day after
    > evaluating the prediction for the current day [6]). For long-term forecasting, we directly
    > forecast 7 days ahead.
    >
    > A.2 Training
    > [...]
    > For electricity-c and traffic-c, we take 500K training windows while for electricity-f and
    > traffic-f, we select 125K and 200K training windows, respectively.
    >
    > A.3 Evaluation
    > Following the experimental settings in [6], one week data from 9/1/2014 00:00 (included)
    > on electricity-c and 6/15/2008 17:00 (included) on traffic-c is left as test sets.
    > For electricity-f and traffic-f dataset, one week data from 8/31/2014 00:15 (included) and
    > 6/15/2008 17:00 (included) is left as test sets, respectively.

    ### Test Metrics

    The reported metrics are the quantile losses $R₀.₅$ and $R₀.₉$.

    ### Results

    > Table 1: Results summary (R₀,₅/R₀,₉ -loss) of all methods. e-c and t-c represent
    > electricity-c and traffic-c, respectively. In the 1st and 3rd row, we perform rolling-day
    > prediction of 7 days while in the 2nd and 4th row, we directly forecast 7 days ahead.
    > TRMF outputs points predictions, so we only report R₀,₅.

    |      | ARIMA       | ETS         | TRMF       | DeepAR      | DeepState   | Ours        |
    |------|-------------|-------------|------------|-------------|-------------|-------------|
    | e-c₁ | 0.154/0.102 | 0.101/0.077 | 0.084/---- | 0.075/0.040 | 0.083/0.056 | 0.059/0.034 |
    | e-c₇ | 0.283/0.109 | 0.121/0.101 | 0.087/---- | 0.082/0.053 | 0.085/0.052 | 0.070/0.044 |
    | t-c₁ | 0.223/0.137 | 0.236/0.148 | 0.186/---- | 0.161/0.099 | 0.167/0.113 | 0.122/0.081 |
    | t-c₇ | 0.492/0.280 | 0.509/0.529 | 0.202/---- | 0.179/0.105 | 0.168/0.114 | 0.139/0.094 |

    Fine (-f)

    |        | electricity-f₁ | traffic-f₁  |
    |--------|----------------|-------------|
    | DeepAR | 0.082/0.063    | 0.230/0.150 |
    | Ours   | 0.074/0.042    | 0.139/0.090 |
    """
