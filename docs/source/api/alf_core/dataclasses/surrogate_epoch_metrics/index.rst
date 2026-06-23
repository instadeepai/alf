Surrogate Epoch Metrics
=======================

The :class:`~alf_core.dataclasses.surrogate_epoch_metrics.SurrogateEpochMetrics` dataclass records training statistics for a single epoch of
surrogate model training, such as loss values and learning rate. These are produced by model
implementations (e.g. :class:`~alf_tools.models.mlp.MLPModel`) during training and can be used for diagnostics and
early stopping decisions.

.. automodule:: alf_core.dataclasses.surrogate_epoch_metrics
   :members:
   :show-inheritance:
   :undoc-members:
