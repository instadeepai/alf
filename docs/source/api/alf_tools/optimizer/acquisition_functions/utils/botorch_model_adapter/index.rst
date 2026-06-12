BoTorch Model Adapter
=====================

Universal adapter that lets both native BoTorch ``Model`` instances and ALF ``BaseModel``
instances work with any BoTorch acquisition function. The adapter detects the model type
automatically: native BoTorch models are passed through directly; ALF models have their
``predict()`` output translated into a BoTorch ``Posterior``. Models that do not provide
prediction variances (e.g. deterministic or CNN-based models) cannot be used with
uncertainty-driven acquisition functions and will raise a ``ValueError`` at call time.

.. automodule:: alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter
   :members:
   :show-inheritance:
   :undoc-members:
