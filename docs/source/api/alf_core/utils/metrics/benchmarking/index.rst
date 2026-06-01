Benchmarking Metrics
====================

Registry and decorator for campaign-level benchmarking metrics that assess end-to-end
optimisation quality — i.e. how well the acquisition strategy finds high-fitness candidates —
as opposed to surrogate model accuracy metrics in ``regression``.

Use ``@register_benchmarking_metric`` to add new functions to ``benchmarking_metric_registry``.
The decorated function must accept at least ``(means, targets)`` and return a ``dict[str, float]``.

.. automodule:: alf_core.utils.metrics.benchmarking
   :members:
   :show-inheritance:
   :undoc-members:
