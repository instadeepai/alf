Metrics Base
============

Shared validators, decorator factories, and registry classes used across all metric domains.
:func:`~alf_core.utils.metrics.base.check_inputs` and :func:`~alf_core.utils.metrics.base.check_variance_validity` enforce array compatibility before any computation.
:func:`~alf_core.utils.metrics.base.require_min_samples` short-circuits metrics when the batch is too small.
:class:`~alf_core.utils.metrics.base.RegressionMetricRegistry` and :class:`~alf_core.utils.metrics.base.ClassificationMetricRegistry` hold the global metric registries
populated at import time by the regression and classification modules.

.. automodule:: alf_core.utils.metrics.base
   :members:
   :show-inheritance:
   :undoc-members:
