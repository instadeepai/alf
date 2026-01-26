Design Task
===========

The ``DesignTask`` runs a multi-round active learning optimization loop. In each round, the surrogate
model is trained on the current training data, the optimizer selects promising candidates, the oracle
evaluates them, and the newly labelled candidates are added to the training set. This iterative process
continues for a specified number of acquisition rounds.

Module
------

alf_core.tasks.design_task
---------------------------

.. automodule:: alf_core.tasks.design_task
   :members:
   :show-inheritance:
   :undoc-members:
