Calibration Metrics
===================

Uncertainty-calibration metrics and reliability-diagram helpers for surrogate model
predictions. ``expected_calibration_error`` and ``rank_expected_calibration_error`` are
registered in ``regression_metric_registry`` via ``@register_requires_variance``.
``calibration_curve`` is a standalone helper that returns raw
``(expected_coverage, observed_coverage)`` arrays for plotting reliability diagrams;
it accepts per-candidate means, variances, and targets.

.. automodule:: alf_core.utils.metrics.calibration
   :members:
   :show-inheritance:
   :undoc-members:
