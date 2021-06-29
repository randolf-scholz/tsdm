r"""
Datasets
========

Basic Usage
-----------

.. code-block:: python

   from tsdm.datasets import Electricity

   print(vars(Electricity))
   Electricity.download()
   Electricity.preprocess()
   x = Electricity.load()

   # or, simply:
   x = Electricity.dataset

Examples
--------
"""

import logging
import subprocess
from abc import ABCMeta, abstractmethod
from functools import cache
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from pandas import DataFrame, Series
from xarray import Dataset, DataArray

from tsdm.config import RAWDATADIR, DATASETDIR

logger = logging.getLogger(__name__)


class DatasetMetaClass(ABCMeta):
    r"""This metaclasses purpose is that any dataset class has certain attributes like
    `rawdata_path`, even before being initialized. As a consequence, the dataset classes
    generally do not need to be initialized.

    Attributes
    ----------
    url: str
        a http address from where the dataset can be downloaded
    dataset: Series, DataFrame, DataArray or Dataset
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
    dataset:      Union[Series, DataFrame, DataArray, Dataset]
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    def __init__(cls, *args, **kwargs):
        """Initializing the paths such that every dataset class has them available,
        even before being instantiated.
        """
        super().__init__(*args, **kwargs)
        cls.rawdata_path = RAWDATADIR.joinpath(cls.__name__)
        cls.rawdata_path.mkdir(parents=True, exist_ok=True)
        cls.dataset_path = DATASETDIR
        cls.dataset_file = DATASETDIR.joinpath(F"{cls.__name__}.h5")
        cls.dataset_path.mkdir(parents=True, exist_ok=True)


class BaseDataset(metaclass=DatasetMetaClass):
    r"""BaseDataset dataset this class implements methods that are available
    for all dataset classes

    Parameters
    ----------
    url: str
        http(s) to download the dataset from

    Attributes
    ----------
    url: str
        a http address from where the dataset can be downloaded
    dataset: Series, DataFrame, DataArray, Dataset
        internal storage of the dataset
    rawdata_path: Path
        location where the raw data is stored
    dataset_path: Path
        location where the pre-processed data is stored
    dataset_file: Path
    """

    url:          str
    # dataset:      Union[Series, DataFrame, DataArray, Dataset]
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

    # Abstract Methods - these MUST be implemented for any dataset subclass

    @classmethod
    @abstractmethod
    def clean(cls):
        """Cleans an already downloaded raw dataset and stores it in hdf5 format
        under the path specified in ``dataset_file``, which, by default is

        .. code-block:: python

            cls.dataset_file = DATASETDIR.joinpath(F"{cls.__name__}.h5")

        .. warning::
            Must be implemented for any dataset class!!
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls):
        """Loads the dataset stored in hdf5 format in the path `cls.dataset_file`.
        Use the following template for dataset classes:

        .. code-block:: python

                @classmethod
                def load(cls):
                    super().load()  # <- makes sure DS is downloaded and preprocessed
                    ...
                    return dataset

        .. warning::
            Must be implemented for any dataset class!!
        """
        if not cls.dataset_file.exists():
            cls.download()
            cls.clean()

    # other class methods

    # todo: remove "type ignore" once fixed in mypy
    # noinspection PyPropertyDefinition
    @classmethod  # type: ignore
    @property
    @cache
    def dataset(cls):
        """Caches the dataset on first execution"""
        return cls.load()

    @classmethod
    def download(cls):
        """Downloads the dataset and stores it in `cls.rawdata_path`.
        The default downloader checks if


        - The url points to kaggle.com => uses `kaggle competition download`
        - The url points to github.com => checkout directory with `svn`
        - Else simply use `wget` to download the `cls.url` content,


        Overwrite if you need custom downloader
        """
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
