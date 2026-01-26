Acquisition Function
====================

The ``AcquisitionFunction`` base class defines the interface for scoring candidates during the
active learning loop. Acquisition functions evaluate which candidates are most promising to query
next by combining predictions and uncertainties from the surrogate model. Examples include
Upper Confidence Bound (UCB), Expected Improvement (EI), and Thompson Sampling.

Module
------

alf_core.optimizer.acquisition_function
----------------------------------------

.. automodule:: alf_core.optimizer.acquisition_function
   :members:
   :show-inheritance:
   :undoc-members:
