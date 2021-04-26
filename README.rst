README
================================================

**T**\ ime **S**\ eries **D**\ atasets and **M**\ odels

This repositry contains tools to import important time series datasets and baseline models

Usage guide
-----------

Installation guide
------------------

1. Create empty environment

.. code-block:: bash

    conda create --name kiwi

2. Set up the channel priorities - we will use `conda-forge` as default

.. code-block:: bash

    conda activate kiwi
    conda config --env --add channels conda-forge
    conda config --env --add channels pytorch
    conda config --env --show channels
    conda config --env --remove channels defaults
    conda config --env --show channels
    conda config --env --set channel_priority strict
    conda config --env --show channel_priority

3. Install the required packages

.. code-block:: bash

    conda env update -f requirements.yaml

4. Install extra packages via pip

.. code-block:: bash

    pip install --upgrade jax jaxlib==0.1.65+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html
    pip install --upgrade tensorflow-gpu

5. Install the `tsdm` package via

.. code-block:: bash

    pip install -e .


