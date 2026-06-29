Random Selection
================

Random Selection is the baseline acquisition function: it scores every candidate uniformly at
random, ignoring the surrogate entirely. It is used to demonstrate that an informed acquisition
function genuinely outperforms uninformed selection. The RNG is seeded per round so draws are
reproducible yet decorrelated across acquisition rounds.

.. automodule:: alf_tools.optimizer.acquisition_functions.random_selection
   :members:
   :show-inheritance:
   :undoc-members:
