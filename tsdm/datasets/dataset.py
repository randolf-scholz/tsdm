r"""
Datasets
========

Basic Usage
-----------

.. code-block:: python

   from tsdm.datasets import Electricity

   print(vars(Electricity))
   Electricity.download()
   ELectricity.preprocess()
   x = Electricity.load()

   # or, simply:
   x = Electricity.dataset

Examples
--------
"""

import logging
import subprocess
from functools import cache
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from pandas import DataFrame, Series
from xarray import Dataset, DataArray
from tsdm.config import RAWDATADIR, DATASETDIR

logger = logging.getLogger(__name__)


class DatasetMetaClass(type):
    r"""This metaclasses purpose is that any dataset class has certain attributes like
    `rawdata_path`, even before being initialized. As a consequence, the dataset classes
    generally do not need to be initialized.

    Attributes
    ----------
    url: str
        a http address from where the dataset can be downloaded
    dataset: Union[Series, DataFrame, DataArray, Dataset]
        internal storage of the dataset
    rawdata_path: Path
        location where the raw data is stored
    dataset_path: Path
        location where the pre-processed data is stored
    dataset_file: Path
    """
    # pylint: disable=no-value-for-parameter
    # see https://stackoverflow.com/q/47615318/9318372

    url:          str
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.rawdata_path = RAWDATADIR.joinpath(cls.__name__)
        cls.rawdata_path.mkdir(parents=True, exist_ok=True)
        cls.dataset_path = DATASETDIR
        cls.dataset_file = DATASETDIR.joinpath(F"{cls.__name__}.h5")
        cls.dataset_path.mkdir(parents=True, exist_ok=True)

    @property
    @cache
    def dataset(cls) -> Union[Series, DataFrame, DataArray, Dataset]:
        """Caches the dataset on first execution"""
        return cls.load()

    def download(cls):
        """Download the dataset"""
        raise NotImplementedError

    def load(cls) -> Union[Series, DataFrame, DataArray, Dataset]:
        """Load the dataset"""
        raise NotImplementedError

    def preprocess(cls):
        """Preprocess the dataset"""
        raise NotImplementedError

    def to_dataloader(cls):
        """Create dataloader from disk"""
        raise NotImplementedError

    def clean(cls):
        """Clean the dataset"""
        raise NotImplementedError


class BaseDataset(metaclass=DatasetMetaClass):
    r"""BaseDataset dataset

    Parameters
    ----------
    url: str
        http(s) to download the dataset from

    Attributes
    ----------
    url: str
        a http address from where the dataset can be downloaded
    dataset: Union[Series, DataFrame, DataArray, Dataset]
        internal storage of the dataset
    rawdata_path: Path
        location where the raw data is stored
    dataset_path: Path
        location where the pre-processed data is stored
    """
    url:          str
    dataset:      Union[Series, DataFrame, DataArray, Dataset]
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    def __init__(self, url: str):
        """Reinitialize dataset from another source

        Parameters
        ----------
        url: str
            http(s) to download the dataset from
        """
        super().__init__()
        self.url = url

    def __new__(cls, *args, **kwargs):
        cls.dataset_path = RAWDATADIR.joinpath(cls.__name__)
        cls.dataset_path.mkdir(parents=True, exist_ok=True)
        cls.rawdata_path = RAWDATADIR.joinpath(cls.__name__)
        cls.rawdata_path.mkdir(parents=True, exist_ok=True)
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def download(cls):
        """Default dataset download. Overwrite if you need custom downloader"""
        dataset = cls.__name__
        parsed_url = urlparse(cls.url)
        logger.info("Obtaining dataset '%s' from %s", dataset, cls.url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                F"kaggle competitions download -p {cls.rawdata_path} -c {kaggle_name}",
                shell=True, check=True)
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                F"svn export {cls.url.replace('tree/master', 'trunk')} {cls.rawdata_path}",
                shell=True, check=True)
        else:  # default parsing, including for UCI datasets
            cut_dirs = cls.url.count("/") - 3
            subprocess.run(
                F"wget -r -np -nH -N --cut-dirs {cut_dirs} -P '{cls.rawdata_path}' {cls.url}",
                shell=True, check=True)

        logger.info("Finished importing dataset '%s' from %s", dataset, cls.url)

    @classmethod
    def load(cls):
        """loads the dataset"""
        if not cls.dataset_file.exists():
            cls.download()
            cls.clean()
        return cls.dataset_file
