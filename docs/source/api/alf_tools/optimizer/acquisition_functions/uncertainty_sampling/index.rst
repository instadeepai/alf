Uncertainty Sampling
====================

Uncertainty Sampling scores each candidate by the surrogate's predictive uncertainty alone,
ignoring the predicted mean. It is the pure-exploration counterpart to
:doc:`Greedy <../greedy/index>` (pure exploitation), and is equivalent to
:doc:`UCB <../ucb/index>` in the limit of a very large exploration parameter. Prefer it when
the goal is to improve the surrogate itself (model-quality metrics such as test RMSE) rather
than to find high-scoring candidates.

Uncertainty is read from ``variances`` when the surrogate reports it. For an ensemble surrogate
that only reports per-member predictions (``empirical_dist``), the disagreement between members
is used instead.

.. automodule:: alf_tools.optimizer.acquisition_functions.uncertainty_sampling
   :members:
   :show-inheritance:
   :undoc-members:
