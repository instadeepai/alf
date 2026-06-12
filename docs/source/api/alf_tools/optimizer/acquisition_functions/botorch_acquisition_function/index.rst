BoTorch Acquisition Functions
=============================

Wraps BoTorch acquisition functions for use in the ALF active learning loop.
Accepts either a native BoTorch ``Model`` or an ALF ``BaseModel``; a
:class:`~alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter.BoTorchModelAdapter`
is inserted automatically when needed.

Configure via :class:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.BotorchAcquisitionConfig`
(specifying the ``name`` and any required ``kwargs``) and pass to
:class:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.BotorchAcquisitionFunction`,
which implements the ALF ``AcquisitionFunction`` interface.  Supported names are
listed in :data:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.ACQUISITION_REGISTRY`:

- ``log_expected_improvement``
- ``upper_confidence_bound``
- ``probability_of_improvement``
- ``log_noisy_expected_improvement``

The ``log_*`` variants score in log-space: values are typically negative and
higher (less negative) means more promising. Argmax selection matches the
non-log acquisition, but absolute scores are not comparable to classic EI
values (such as ALF's own ``ExpectedImprovement``).

.. automodule:: alf_tools.optimizer.acquisition_functions.botorch_acquisition_function
   :members:
   :show-inheritance:
   :undoc-members:
