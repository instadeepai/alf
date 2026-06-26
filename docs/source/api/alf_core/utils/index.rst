Utils
=====

Utility modules provide enumeration types, evaluation metrics, and state logging functionality.
The :mod:`alf_core.utils.enums` module defines :class:`~alf_core.utils.enums.ProblemType` for specifying regression or classification tasks.
The ``metrics`` module contains functions for computing regression accuracy, uncertainty
calibration, and classification metrics. The :mod:`alf_core.utils.state_logger` module provides classes for
logging and tracking the state of active learning tasks during execution.

.. toctree::
   :maxdepth: 1

   Enums <enums/index>
   Metrics <metrics/index>
   State Logger <state_logger/index>
