Random
======

Random is the mandatory "floor" baseline: it assigns each candidate a uniform random score,
completely ignoring the surrogate's predictions (mean or uncertainty). Any acquisition strategy
that does not clearly beat Random is not adding value over blind selection, which is why every
acquisition-strategy comparison should include it.

Scores are drawn from a per-round RNG seeded by ``(seed, round)``, so results are reproducible
for a given seed while decorrelating across rounds.

.. automodule:: alf_tools.optimizer.acquisition_functions.random_acquisition
   :members:
   :show-inheritance:
   :undoc-members:
