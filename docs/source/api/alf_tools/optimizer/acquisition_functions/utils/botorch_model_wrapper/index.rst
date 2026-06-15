BoTorch Model Wrapper
=====================

Universal wrapper that lets both native BoTorch ``Model`` instances and ALF ``BaseModel``
instances work with any BoTorch acquisition function. The wrapper detects the model type
automatically and exposes a ``provides_joint_posterior`` capability flag. Native BoTorch
models are passed through directly. ALF models that expose a trained ``botorch_model``
(e.g. ``GPModel``) are joint-capable: ``posterior()`` is delegated to that inner model,
yielding a true joint covariance and honouring ``observation_noise``/``posterior_transform``/
``output_indices``. Marginal-only ALF models (``predict()``-based, no ``botorch_model``) have
their ``predict()`` output translated into a diagonal (per-point independent) BoTorch
``Posterior`` and report the single-output defaults ``num_outputs=1`` / ``batch_shape=torch.Size([])``.
Models that do not provide prediction variances (e.g. deterministic or CNN-based models) cannot be
used with uncertainty-driven acquisition functions and will raise a ``ValueError`` at call time.

The module also exposes ``resolve_botorch_model``, the shared capability check used across the
BoTorch integration: it returns a surrogate's joint-posterior BoTorch model (native model, or an
ALF model's trained ``botorch_model``) or ``None`` for marginal-only models.

.. automodule:: alf_tools.optimizer.acquisition_functions.utils.botorch_model_wrapper
   :members:
   :show-inheritance:
   :undoc-members:
