CoreSet
=======

CoreSet is a diversity-based acquisition function that selects candidates by maximising
coverage of the input space using greedy k-centres. Rather than relying on uncertainty
estimates, it calls `surrogate.featurise()` to obtain embeddings, then iteratively picks
the candidate most distant from the current training set and previously selected candidates.
This makes it compatible with any model and independent of uncertainty calibration.

Candidates are scored by selection rank (`n_select − step`), so the first chosen candidate
receives the highest score; unselected candidates receive 0. The model's `featurise()`
method must return a 2-D array of shape `(n_inputs, d)`.

.. automodule:: alf_tools.optimizer.acquisition_functions.core_set
   :members:
   :show-inheritance:
   :undoc-members:
