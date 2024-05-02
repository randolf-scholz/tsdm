r"""Datasets from the PhysioNet project.

References:
    - `PhysioNet <https://physionet.org/>`_
    - Challenge 2012 <https://physionet.org/content/challenge-2012>_
    - Challenge 2019 <https://physionet.org/content/challenge-2019>
"""

__all__ = [
    "PhysioNet2012",
    "PhysioNet2019",
]

from tsdm.datasets.physionet.physionet2012 import PhysioNet2012
from tsdm.datasets.physionet.physionet2019 import PhysioNet2019
