BoTorch Model Wrapper
=====================

Universal wrapper that lets both native BoTorch ``Model`` instances and ALF ``BaseModel``
instances work with any BoTorch acquisition function. The wrapper detects the model type
automatically: native BoTorch models are passed through directly; ALF models have their
``predict()`` output translated into a BoTorch ``Posterior``. Models that do not provide
prediction variances (e.g. deterministic or CNN-based models) cannot be used with
uncertainty-driven acquisition functions and will raise a ``ValueError`` at call time.

.. automodule:: alf_tools.optimizer.acquisition_functions.utils.botorch_model_wrapper
   :members:
   :show-inheritance:
   :undoc-members:
