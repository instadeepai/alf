Botorch Continuous Search
=========================

``BotorchContinuousSearch`` is a search function for continuous optimisation with BoTorch.
It returns an empty candidate list, which signals ``BoTorchAcquisition`` to generate
candidates by optimising the acquisition function directly in continuous space
(via ``optimize_acqf``) rather than scoring a discrete pool.

Use it with continuous search spaces where gradient-based candidate generation is
desired; the acquisition function must be constructed with ``bounds``.

.. automodule:: alf_tools.optimizer.search.botorch_continuous_search
   :members:
   :show-inheritance:
   :undoc-members:
