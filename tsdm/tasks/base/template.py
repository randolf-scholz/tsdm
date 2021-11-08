r"""Example Task Implementation.

TODO: Module description
"""

from __future__ import annotations

import logging

# __all__ = []


__logger__ = logging.getLogger(__name__)


class ExampleTask:
    r"""Example of how a Task should be documented / implemented.

    Make sure to add the following 4 sections (exact names!) ``Paper``, ``Evaluation Protocol``,
    ``Test-Metric`` and ``Results``

    Paper
    -----

    Copy-paste paper tile, authors, venue and a hyperlink. For example:


    Evaluation Protocol
    -------------------

    Copy paste appropriate passages from the paper outlining the evaluation protocol.
    Use 4 spaces to use restructuredText block quotes, for example:

        A.1 Datasets and Evaluation Criteria
        [...]
        traffic 4 : A collection of 15 months of daily data from the California Department of
        Transportation. The data describes the occupancy rate, between 0 and 1, of different car
        lanes of San Francisco bay area freeways. The data was sampled every 10 minutes, and we
        again aggregate the columns to obtain hourly traffic data to finally get n = 963,
        T = 10, 560. The coefficient of variation for traffic is 0.8565.

    Test-Metric
    -----------

    Write or quote what was the test metric used in the paper. For example:

    **Normalized deviation (ND)**

        .. math::
            𝖭𝖣(Y, Ŷ) = \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t)∈Ω_\text{test}}|Ŷ_{it}-Y_{it}|\Big)
            \Big/ \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Y_{it}|\Big)

    Results
    -------

    Here you can paste a table from the paper. The onle tool https://www.tablesgenerator.com/ is
    your biggest friend. Use the `File>Paste Table Data` option to easily import tables.
    For output, select `Text` and check the `Use reStructuredText syntax` checkbox. For example:

    +-------+-------+-------------+-------------+---------------+
    | Model | TRMF  | N-BEATS (G) | N-BEATS (I) | N-BEATS (I+G) |
    +=======+=======+=============+=============+===============+
    | ND    | 0.187 | 0.112       | 0.110       | 0.111         |
    +-------+-------+-------------+-------------+---------------+
    """
