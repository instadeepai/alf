Supervised Task
===============

The :class:`~alf_core.tasks.supervised_task.SupervisedTask` trains and evaluates models on fixed data splits without active learning. It
trains the surrogate model on the training set, evaluates predictions on validation and test sets, and
computes evaluation metrics. This task is useful for benchmarking model performance and validating
surrogate model quality.

.. automodule:: alf_core.tasks.supervised_task
   :members:
   :show-inheritance:
   :undoc-members:
