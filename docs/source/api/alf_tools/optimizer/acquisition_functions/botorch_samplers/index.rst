BoTorch MC Samplers
===================

``BoTorchMCSampler`` is a configuration wrapper around BoTorch's Monte Carlo
samplers, used by Monte Carlo acquisition functions (qEI, qNEI, qUCB) to
approximate expectations over the posterior. It supports Sobol quasi-Monte Carlo
sampling (``"sobol"``, recommended for better coverage) and independent sampling
(``"iid"``), with a configurable sample count and optional seed for reproducibility.

It is not a search function: pass it to ``BoTorchAcquisition`` via the ``sampler``
argument to control how acquisition values are estimated.

.. automodule:: alf_tools.optimizer.acquisition_functions.botorch_samplers
   :members:
   :show-inheritance:
   :undoc-members:
