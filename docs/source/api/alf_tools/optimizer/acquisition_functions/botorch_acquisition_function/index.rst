BoTorch Acquisition Functions
=============================

Wraps BoTorch acquisition functions for use in the ALF active learning loop.
Accepts either a native BoTorch ``Model`` or an ALF ``BaseModel``; the adapter is
inserted automatically so callers do not need to handle the conversion.

Provides four ready-to-use functional factories —
:func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.expected_improvement`,
:func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.upper_confidence_bound`,
:func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.probability_of_improvement`,
and :func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.log_noisy_expected_improvement`
— along with the :data:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.ACQUISITION_REGISTRY`
mapping and a :class:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.BotorchAcquisitionFunction`
class that selects a factory by name via :class:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.BotorchAcquisitionConfig`.

.. automodule:: alf_tools.optimizer.acquisition_functions.botorch_acquisition_function
   :members:
   :show-inheritance:
   :undoc-members:
