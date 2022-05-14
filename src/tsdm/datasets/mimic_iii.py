r"""MIMIC-III Clinical Database.

Abstract
--------

MIMIC-III is a large, freely-available database comprising de-identified health-related
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

__all__ = ["MIMIC_III"]

import os
import subprocess
from getpass import getpass
from pathlib import Path
from typing import Optional, Union

from tsdm.datasets.base import SimpleDataset


class MIMIC_III(SimpleDataset):
    r"""MIMIC-III SimpleDataset Database."""

    base_url: str = r"https://physionet.org/files/mimiciii/1.4/"
    r"""HTTP address from where the dataset can be downloaded."""
    info_url: str = r"https://physionet.org/content/mimiciii/1.4/"
    r"""HTTP address containing additional information about the dataset."""
    rawdata_files = "MIMIC-III.zip"

    def _clean(self) -> None:
        r"""Clean an already downloaded raw dataset and stores it in hdf5 format."""

    def _load(self):
        r"""Load the dataset stored in hdf5 format in the path `cls.dataset_files`."""

    def _download(self, *, url: Optional[Union[str, Path]] = None) -> None:
        r"""Download the dataset and stores it in `cls.rawdata_dir`.

        The default downloader checks if

        1. The url points to kaggle.com => uses `kaggle competition download`
        2. The url points to github.com => checkout directory with `svn`
        3. Else simply use `wget` to download the `cls.url` content,

        Overwrite if you need custom downloader

        Parameters
        ----------
        url: Optional[Union[str, Path]], default None
        """
        if self.url is None:
            self.__logger__.info("Dataset provides no url. Assumed offline")
            return

        self.__logger__.info("Obtaining dataset from %s", self.url)

        cut_dirs = self.url.count("/") - 3

        user = input("MIMIC-III username: ")
        password = getpass(prompt="MIMIC-III password: ", stream=None)

        os.environ["PASSWORD"] = password

        url = self.url if url is None else url

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            f"--cut-dirs {cut_dirs} -P '{self.rawdata_dir}' {url}",
            shell=True,
            check=True,
        )

        self.__logger__.info("Finished importing dataset from %s", self.url)
