Model Utilities
===============

Shared utilities for ALF model implementations.

:func:`~alf_tools.models.utils.data_utils.transform_data` (in :mod:`alf_tools.models.utils.data_utils`) applies input normalisation and output standardisation
to training data in a single call, returning the transformed tensors and fitted normaliser
objects for later use at inference time.

:class:`~alf_tools.models.utils.botorch_model_wrapper.BotorchModelWrapper` bridges ALF's :class:`~alf_core.model.base_model.BaseModel` interface and native BoTorch models to
BoTorch acquisition functions, automatically routing through either the BoTorch :meth:`~alf_tools.models.utils.botorch_model_wrapper.BotorchModelWrapper.posterior`
method or ALF's ``predict()`` depending on the model type.

:func:`~alf_tools.models.utils.config_utils.build_from_target` (in :mod:`alf_tools.models.utils.config_utils`) instantiates any GPyTorch prior or constraint
from a serialisable ``_target_`` config dict, enabling fully configuration-driven kernel setup.

The :mod:`alf_tools.models.utils.botorch_utils` module provides conversion utilities between ALF's data structures
(``Candidate``, :class:`~alf_core.dataclasses.predictions.Predictions`) and BoTorch's expected formats (tensors, posteriors).

.. automodule:: alf_tools.models.utils.data_utils
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: alf_tools.models.utils.botorch_model_wrapper
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: alf_tools.models.utils.config_utils
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: alf_tools.models.utils.botorch_utils
   :members:
   :show-inheritance:
   :undoc-members:
