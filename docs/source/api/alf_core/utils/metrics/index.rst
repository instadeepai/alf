Metrics
=======

The ``metrics`` package provides evaluation metrics for assessing surrogate model and campaign
performance. It is organised into four sub-modules: shared validators and registries (``base``),
regression and calibration metrics (``regression``), classification metrics (``classification``),
and campaign-level benchmarking metrics (``benchmarking``).

All metrics are registered at import time and accessible via the global registry instances
exported from the top-level ``alf_core.utils.metrics`` namespace.

.. toctree::
   :maxdepth: 1

   Base <base/index>
   Benchmarking <benchmarking/index>
   Classification <classification/index>
   Regression <regression/index>
