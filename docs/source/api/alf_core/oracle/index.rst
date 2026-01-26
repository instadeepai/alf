Oracle
======

The ``Oracle`` provides ground-truth labels for candidates. It can operate in two modes: offline
(using a dataset to query labels) or online (using a model to generate labels, simulating real
experiments). The oracle is used during the active learning loop to evaluate selected candidates
and provide feedback for training the surrogate model.

Module
------

alf_core.oracle.oracle
-----------------------

.. automodule:: alf_core.oracle.oracle
   :members:
   :show-inheritance:
   :undoc-members:
