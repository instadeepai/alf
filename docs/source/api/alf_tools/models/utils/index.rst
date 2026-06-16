Model Utilities
===============

Shared utilities for ALF model implementations.

``transform_data`` (in ``data_utils``) applies input normalisation and output standardisation
to training data in a single call, returning the transformed tensors and fitted normaliser
objects for later use at inference time.

``BotorchModelWrapper`` bridges ALF's ``BaseModel`` interface and native BoTorch models to
BoTorch acquisition functions, automatically routing through either the BoTorch ``posterior()``
method or ALF's ``predict()`` depending on the model type.

``build_from_target`` (in ``config_utils``) instantiates any GPyTorch prior or constraint
from a serialisable ``_target_`` config dict, enabling fully configuration-driven kernel setup.

The ``botorch_utils`` module provides conversion utilities between ALF's data structures
(``Candidate``, ``Predictions``) and BoTorch's expected formats (tensors, posteriors).

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
