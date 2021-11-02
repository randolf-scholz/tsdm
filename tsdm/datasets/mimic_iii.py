r"""MIMIC-III Clinical Database.

Abstract
--------

MIMIC-III is a large, freely-available database comprising deidentified health-related
data associated with over forty thousand patients who stayed in critical care units of
the Beth Israel Deaconess Medical Center between 2001 and 2012.
The database includes information such as demographics, vital sign measurements made at
the bedside (~1 data point per hour), laboratory test results, procedures, medications,
caregiver notes, imaging reports, and mortality (including post-hospital discharge).

MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical
decision-rule improvement, and electronic tool development. It is notable for three
factors: it is freely available to researchers worldwide; it encompasses a diverse and
very large population of ICU patients; and it contains highly granular data, including
vital signs, laboratory results, and medications.
"""


from __future__ import annotations

__all__ = ["MIMIC_III"]

import logging
import os
import subprocess
from getpass import getpass

from tsdm.datasets.dataset import BaseDataset

# from pathlib import Path
# from typing import Final
# from zipfile import ZipFile
#
# import numpy as np
# from pandas import DataFrame


__logger__ = logging.getLogger(__name__)


class MIMIC_III(BaseDataset):
    r"""MIMIC-III Clinical Database."""

    url: str = r"https://physionet.org/files/mimiciii/1.4/"
    info_url: str = r"https://physionet.org/content/mimiciii/1.4/"

    @classmethod
    def download(cls):
        r"""Download the dataset and stores it in `cls.rawdata_path`.

        The default downloader checks if

        1. The url points to kaggle.com => uses `kaggle competition download`
        2. The url points to github.com => checkout directory with `svn`
        3. Else simply use `wget` to download the `cls.url` content,

        Overwrite if you need custom downloader
        """
        if cls.url is None:
            __logger__.info(
                "Dataset '%s' provides no url. Assumed offline", cls.__name__
            )
            return

        dataset = cls.__name__
        __logger__.info("Obtaining dataset '%s' from %s", dataset, cls.url)

        cut_dirs = cls.url.count("/") - 3

        user = input("MIMIC-III username")
        password = getpass(prompt="MIMIC-III password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            f"--cut-dirs {cut_dirs} -P '{cls.rawdata_path}' {cls.url}",
            shell=True,
            check=True,
        )

        __logger__.info("Finished importing dataset '%s' from %s", dataset, cls.url)
