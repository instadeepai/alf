Metrics Base
============

Shared validators, decorator factories, and registry classes used across all metric domains.
``check_inputs`` and ``check_variance_validity`` enforce array compatibility before any computation.
``require_min_samples`` short-circuits metrics when the batch is too small.
``RegressionMetricRegistry`` and ``ClassificationMetricRegistry`` hold the global metric registries
populated at import time by the regression and classification modules.

.. automodule:: alf_core.utils.metrics.base
   :members:
   :show-inheritance:
   :undoc-members:
