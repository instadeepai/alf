Regression Metrics
==================

Regression and uncertainty-calibration metrics for evaluating surrogate model predictions.
All functions are registered in ``regression_metric_registry`` at import time via
``@register_requires_variance`` (for metrics that need uncertainty estimates) or
``@register_no_variance_required`` (for point-prediction metrics).

Accuracy metrics (no variance required): MSE, Spearman, Pearson, pairwise cross-entropy.

Calibration and UQ metrics (variance required): expected calibration error, rank ECE,
coverage, rank coverage, interval width, rank width, residual Spearman, residual Pearson,
UCB regret, and UCB regret sweep.

.. automodule:: alf_core.utils.metrics.regression
   :members:
   :show-inheritance:
   :undoc-members:
