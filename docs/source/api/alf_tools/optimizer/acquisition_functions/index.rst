Acquisition Functions
=====================

Concrete acquisition function implementations that inherit from ``alf_core.optimizer.AcquisitionFunction``.
These functions score candidates during the active learning loop to determine which candidates are most
promising for evaluation.

.. toctree::
   :maxdepth: 1

   Expected Improvement <expected_improvement/index>
   Greedy <greedy/index>
   Thompson Sampling <thompson_sampling/index>
   Upper Confidence Bound (UCB) <ucb/index>
