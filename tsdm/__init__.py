r"""
Provides
  1. Facility to import some commonly used time series datasets
  2. Facility to import some commonly used time series models
  3. Facility to preprocess time series datasets

Constants
---------

.. data:: HOMEDIR, BASEDIR, LOGDIR, MODELDIR, DATASETDIR, RAWDATADIR

    Paths of the internally used directory tree

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

import tsdm.util as util
import tsdm.datasets as datasets
import tsdm.config as config
import tsdm.losses as loss
import tsdm.models as models

__all__ = ['util', 'datasets', 'models', 'config', 'loss', 'models']
