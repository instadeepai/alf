Optimizer
=========

The ``Optimizer`` class orchestrates the active learning loop by coordinating the acquisition
function and search strategy. It selects the most promising candidates for evaluation by first
obtaining a candidate pool from the search strategy, then scoring candidates using the acquisition
function, and finally selecting the top-k candidates for oracle evaluation.

Module
------

alf_core.optimizer.optimizer
-----------------------------

.. automodule:: alf_core.optimizer.optimizer
   :members:
   :show-inheritance:
   :undoc-members:
