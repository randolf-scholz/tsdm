.. include:: /includes.rst.txt

tsdm
====

**T**\ ime **S**\ eries **D**\ atasets and **M**\ odels

Submodules
----------

.. autosummary::
    :toctree:
    :template: module.rst
    :recursive:

    tsdm

.. toctree::

   readme
   changelog


Example
_______

.. code-block:: python

   import tsdm

   dataset = tsdm.load_dataset('electricity')
   x, m, d = tsdm.make_masked_format('dataset')
   ODE_RNN = tsdm.load_model('ODE-RNN')
   model = ODE_RNN(...)  # see model description for details


Installation
------------

Install the tsdm_ package using ``pip`` by

.. code-block:: bash

   cd tsdm
   pip install -e .

Here we assume that you want to install the package in editable mode, because
you would like to contribute to it. This package is not available on PyPI, it
might be in the future, though.

Contribute
----------

- Issue Tracker: https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm/-/issues
- Source Code: https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm

Support
-------

If you encounter issues, please let us know.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
