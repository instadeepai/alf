Regression Metrics
==================

Regression and uncertainty-calibration metrics for evaluating surrogate model predictions.
All functions are registered in ``regression_metric_registry`` at import time via
:func:`@register_requires_variance <alf_core.utils.metrics.regression.register_requires_variance>` (for metrics that need uncertainty estimates) or
:func:`@register_no_variance_required <alf_core.utils.metrics.regression.register_no_variance_required>` (for point-prediction metrics).

Accuracy metrics (no variance required): MSE, Spearman, Pearson, pairwise cross-entropy,
top-k mean, top-k max, and hit rate.

UQ metrics (variance required): coverage, rank coverage, interval width, rank width,
residual Spearman, residual Pearson, NLL (Gaussian), UCB regret, and UCB regret sweep.
Calibration error metrics live in the :doc:`Calibration <../calibration/index>` module.

.. automodule:: alf_core.utils.metrics.regression
   :members:
   :show-inheritance:
   :undoc-members:
