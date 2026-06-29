Uncertainty-Based
=================

The uncertainty-based (maximum-variance) acquisition function scores each candidate by its
predictive variance alone, ignoring the mean, so the candidates the surrogate is least certain
about are selected first. Unlike the maximising acquisitions, it targets *model-error reduction*
and requires an uncertainty-aware surrogate such as an ensemble or a Gaussian process.

.. automodule:: alf_tools.optimizer.acquisition_functions.uncertainty_based
   :members:
   :show-inheritance:
   :undoc-members:
