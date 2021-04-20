
tsdm
================================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   tsdm
   readme
   changelog


Purpose
_______

Why do we need the package?



Example
_______

To ease the usage this package tries to follow the guidelines of scikit-learn
estimators
https://scikit-learn.org/stable/developers/develop.html. In practise the usage
looks like this:

.. code-block:: python

   import tsdm

   trained_model = tsdm.models.ExponentialDecay().fit(points_in_time, labels)
   trained_model.predict(points_in_time)

Features
--------

The package implements the following methods

- something
- something more

Installation
------------

Install the glucose package using `pip` by

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
