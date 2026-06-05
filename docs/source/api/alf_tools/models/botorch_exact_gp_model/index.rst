BoTorch GP Model
================

A Gaussian Process model built on BoTorch's ``SingleTaskGP``, offering better default
hyperparameter priors (Hvarfner et al. 2024), automatic output standardisation, and seamless
integration with BoTorch acquisition functions.

``BoTorchTrainConfig`` supports two fitting backends: ``'scipy'`` (L-BFGS-B, default) and
``'torch'`` (Adam). Input normalisation and output standardisation are applied automatically
before constructing the underlying BoTorch model. Use the ``botorch_model`` property to
access the trained ``SingleTaskGP`` directly for use with native BoTorch acquisition functions.

.. automodule:: alf_tools.models.botorch_exact_gp_model
   :members:
   :show-inheritance:
   :undoc-members:
