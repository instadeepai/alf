GP Model
========

A Gaussian Process (GP) model implementation for protein sequence prediction.
This model uses a BoTorch ``SingleTaskGP`` backbone with a configurable kernel to
model sequence fitness, returning both predicted means and variances (uncertainty
estimates). The trained backbone is exposed via the ``botorch_model`` property for
use with native BoTorch acquisition functions.

``GPTrainConfig`` defaults to ``normalise_inputs_strategy="minmax"`` and
``standardise_outputs=True``: GP kernels measure distances between inputs, so
min-max scaling to [0, 1] improves marginal log-likelihood optimisation, and
output standardisation improves numerical stability.

.. automodule:: alf_tools.models.gp
   :members:
   :show-inheritance:
   :undoc-members:
