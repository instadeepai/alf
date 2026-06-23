Base Model
==========

Abstract base class and shared training configuration for all ALF models.
:class:`~alf_core.model.base_model.BaseTrainConfig` is the shared dataclass that every concrete training config
(e.g. :class:`~alf_tools.models.gp.GPTrainConfig`, :class:`~alf_tools.models.cnn.CNNTrainConfig`) inherits from. It exposes the
``normalise_inputs_strategy`` and ``standardise_outputs`` fields that control data
transformation at training time.

.. automodule:: alf_core.model.base_model
   :members:
   :show-inheritance:
   :undoc-members:
