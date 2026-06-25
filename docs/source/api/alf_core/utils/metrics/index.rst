Metrics
=======

The ``metrics`` module provides evaluation metrics for assessing model performance, split across two
registries by problem type. The ``regression_metric_registry`` contains accuracy metrics
(e.g., Spearman correlation, MSE, Pearson correlation) and uncertainty calibration metrics
(e.g., calibration error, coverage, width). The ``classification_metric_registry`` contains
metrics for binary and multiclass tasks (accuracy, F1, precision, recall, AUC-ROC).
:class:`~alf_core.dataclasses.results.Results` automatically selects the appropriate registry
based on the dataset's :class:`~alf_core.utils.enums.ProblemType`.

All metrics are registered at import time and accessible via the global registry instances
exported from the top-level ``alf_core.utils.metrics`` namespace. Two further modules sit
outside the registries: acquisition batch metrics, which operate on candidate objects rather
than prediction arrays, and design task metrics, which aggregate per-round values into
end-of-experiment summary scores.

.. toctree::
   :maxdepth: 1

   Acquisition Batch <acquisition_batch/index>
   Base <base/index>
   Calibration <calibration/index>
   Classification <classification/index>
   Design Task Metrics <aggregate/index>
   Regression <regression/index>
