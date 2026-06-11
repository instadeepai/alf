Summary Metrics
===============

Standalone metrics not registered in ``regression_metric_registry``.

``auc_top_k``: normalised area under the top-k mean curve — primary sample-efficiency
ranking metric that accepts per-round aggregated values.  ``calibration_curve``:
returns raw ``(expected_coverage, observed_coverage)`` arrays for plotting reliability
diagrams; accepts per-candidate means, variances, and targets.

.. automodule:: alf_core.utils.metrics.aggregate
   :members:
   :show-inheritance:
   :undoc-members:
