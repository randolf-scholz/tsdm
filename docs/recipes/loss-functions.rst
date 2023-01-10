Loss-Functions
--------------

By default, weight channels inversely proportial to missing rate.

This ensures the model fits on all channels instead of underfitting on sparse
and overfitting on channels with many observations.
Since some channels might be observed all the time, we add a small
constant to the denominator to avoid division by zero.

For a single forecasting window $T$, the loss is:

.. math::
    &∑_{t∈T} ∑_i \frac{[m_{t, i} ?(ŷ_{t, i} - y_{t, i})^2 : 0]}{∑_{t∈T} m_{t, i}}
    \\ &= ∑_{t∈T} ∑_i \frac{1}{|T|} \frac{|T|}{∑_{t∈T} m_{t, i}} [m_{t, i} ?(ŷ_{t, i} - y_{t, i})^2 : 0]
    \\ &≈ ∑_{t∈T} ∑_i \frac{ωᵢ}{|T|}[m_{t, i} ?(ŷ_{t, i} - y_{t, i})^2 : 0]
    \\ &≈ \frac{1}{|T|} ∑_{t∈T} ∑_i ωᵢ [m_{t, i} ?(ŷ_{t, i} - y_{t, i})^2 : 0]

where :math:`ω_i = \frac{1}{\frac{1}{|T|}∑_{t∈T} m_{t, i}}` is the inverse of the observation rate for a given channel. This has the advantage that the loss works when there are no observations for a given channel for a given window. (We assume there is at least a single observation in every channel in the training set).

However, note that this does not work if we compute the loss in a batch with padding, since this will lead to more NaN values than expected.
were we normalize by the number of observed values in the window. Note that in an individual window the number of observations in a channel can be zero, which is an issue.

In order to not have to recompute this normalization all the time, we consider the the averge missing rate in the training set and use this as a normalization constant.

Note that if :math:`∑_{t∈T} m_{t, i} = 0`, then the loss is zero for that channel.

For a long horizon, :math:`w_i = ∑_{t∈T} \frac{m_{t,i}}{|T|}`
