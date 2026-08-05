State Logger
============

The :mod:`alf_core.utils.state_logger` module provides classes for logging and tracking the state of active learning
tasks during execution. Loggers can write task state to different backends (e.g., terminal output, files,
databases) and are called at key points during task execution to record progress and results.

Each logger exposes two entry points: ``log`` records per-round state (metrics, training history,
predictions, and the acquisition batch), while ``log_summary`` records only a set of scalar
end-of-experiment metrics — such as the ``experiment_summary`` emitted by :class:`~alf_core.tasks.design_task.DesignTask` — without
touching round-level state.

For ``FileStateLogger`` these write to separate files so each stays single-schema: ``log`` appends a
row per round to ``metrics.csv`` with an explicit leading ``round`` column, while ``log_summary``
appends summary rows to ``summary.csv`` (which has no ``round`` column, as summary metrics are not
tied to any one acquisition round).

.. automodule:: alf_core.utils.state_logger
   :members:
   :show-inheritance:
   :undoc-members:
