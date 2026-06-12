Classification Metrics
======================

Standard classification metrics for evaluating models that output class probabilities.
All functions are registered in `classification_metric_registry` at import time via
`@register_classification_metric`, which also enforces that `probs` is 2-D and
that batch sizes match before delegating to the metric function.

Included metrics: accuracy, F1 (macro), precision (macro), recall (macro), and AUC-ROC
(binary one-class probability; multiclass one-vs-rest).

.. automodule:: alf_core.utils.metrics.classification
   :members:
   :show-inheritance:
   :undoc-members:
