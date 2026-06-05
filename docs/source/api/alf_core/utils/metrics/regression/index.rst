Regression Metrics
==================

Regression and uncertainty-calibration metrics for evaluating surrogate model predictions.
All functions are registered in ``regression_metric_registry`` at import time via
``@register_requires_variance`` (for metrics that need uncertainty estimates) or
``@register_no_variance_required`` (for point-prediction metrics).

Accuracy metrics (no variance required): MSE, Spearman, Pearson, pairwise cross-entropy,
top-k mean, top-k max, and hit rate.

Calibration and UQ metrics (variance required): expected calibration error, rank ECE,
coverage, rank coverage, interval width, rank width, residual Spearman, residual Pearson,
NLL (Gaussian), UCB regret, and UCB regret sweep.

Campaign-level metrics (standalone, not in registry): ``auc_top_k`` for normalised
area-under-the-top-k-mean-curve sample efficiency scoring, and ``calibration_curve``
for generating reliability diagram arrays.

.. automodule:: alf_core.utils.metrics.regression
   :members:
   :show-inheritance:
   :undoc-members:
