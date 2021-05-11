r"""
Provides
  1. Facility to import some commonly used time series datasets
  2. Facility to import some commonly used time series models
  3. Facility to preprocess time series datasets

Constants
---------

.. data:: HOMEDIR, BASEDIR, LOGDIR, MODELDIR, DATASETDIR, RAWDATADIR

    Paths of the internally used directories

.. data:: AVAILABLE_MODELS

    Set of all available models

.. data:: AVAILABLE_DATASETS

    Set of all available datasets

.. data:: CONFIG

    Dictionary containing basic configuration of TSDM

.. data:: DATASETS

    Dictionary containing sources of the available datasets

.. data:: HASHES

    Dictionary containing hash values for both models and datasets

.. data:: MODELS

    Dictionary containing sources of the available models

Functions
---------
"""

import util
from .config import HOMEDIR, BASEDIR, LOGDIR, MODELDIR, DATASETDIR, RAWDATADIR, \
    AVAILABLE_MODELS, AVAILABLE_DATASETS, MODELS, HASHES, DATASETS, CONFIG
from .converters import make_dense_triplets, make_sparse_triplets, make_masked_format
from .dataset_cleaners import clean_dataset
from .dataset_io import download_dataset
from .dataset_loaders import load_dataset
from .model_cleaners import clean_model
from .model_io import download_model
from .model_loaders import load_model

__all__ = ['HOMEDIR', 'BASEDIR', 'LOGDIR', 'MODELDIR', 'DATASETDIR', 'RAWDATADIR',
           'AVAILABLE_MODELS', 'AVAILABLE_DATASETS', 'MODELS', 'HASHES', 'DATASETS', 'CONFIG',
           'download_model', 'clean_model', 'load_model', 'download_dataset', 'clean_dataset', 'load_dataset',
           'make_dense_triplets', 'make_sparse_triplets', 'make_masked_format']
