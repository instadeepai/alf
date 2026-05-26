Base Model
==========

Abstract base class and shared training configuration for all ALF models.
``BaseTrainConfig`` is the shared dataclass that every concrete training config
(e.g. ``GPTrainConfig``, ``CNNTrainConfig``) inherits from. It exposes the
``normalise_inputs`` and ``standardise_outputs`` flags that control data
transformation at training time.

.. automodule:: alf_core.model.base_model
   :members:
   :show-inheritance:
   :undoc-members:
