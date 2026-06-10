Summary Metrics
===============

Standalone summary metrics that aggregate information across multiple rounds of an
active learning experiment.  Unlike the registered per-round metrics in
``regression.py``, these functions are not registered in ``regression_metric_registry``
and accept multi-round inputs rather than per-candidate arrays.

``auc_top_k``: normalised area under the top-k mean curve — primary sample-efficiency
ranking metric.  ``calibration_curve``: returns raw ``(expected_coverage,
observed_coverage)`` arrays for plotting reliability diagrams.

.. automodule:: alf_core.utils.metrics.summary
   :members:
   :show-inheritance:
   :undoc-members:
