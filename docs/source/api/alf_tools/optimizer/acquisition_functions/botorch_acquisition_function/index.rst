BoTorch Acquisition Functions
=============================

Wraps any BoTorch acquisition function for use in the ALF active learning loop.
Accepts either a native BoTorch ``Model`` or an ALF ``BaseModel``; the adapter is
inserted automatically so callers do not need to handle the conversion.

Includes a Hydra-driven :class:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.BotorchAcquisitionFunction`
for config-file usage, plus four ready-to-use functional factories:
:func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.expected_improvement`,
:func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.upper_confidence_bound`,
:func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.probability_of_improvement`,
and :func:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition_function.log_noisy_expected_improvement`.

.. automodule:: alf_tools.optimizer.acquisition_functions.botorch_acquisition_function
   :members:
   :show-inheritance:
   :undoc-members:
