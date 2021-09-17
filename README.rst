**T**\ ime **S**\ eries **D**\ atasets and **M**\ odels
=======================================================

This repository contains tools to import important time series datasets and baseline models

Installation guide
------------------

.. code-block:: bash

    pip install -e .


Multiple Origins Push
---------------------

The project is located at
 - https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tsdm
 - https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm

To push to both repositories do the following

1. Remove all remotes.

.. code-block:: shell

    git remote -v
    git remote remove ...

2. Add both hildesheim and berlin remote

.. code-block:: shell
    git remote add hildesheim git@software.ismll.uni-hildesheim.de:ISMLL-internal/special-interest-group-time-series/tsdm.git
    git remote add berlin git@git.tu-berlin.de:bvt-htbd/kiwi/tf1/tsdm.git

3. Tell GIT from which remote to perform pulls for the branch

.. code-block:: shell

    git push -u hildesheim master


