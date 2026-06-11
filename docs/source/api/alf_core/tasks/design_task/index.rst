Design Task
===========

The ``DesignTask`` runs a multi-round active learning optimization loop. In each round, the surrogate
model is trained on the current training data, the optimizer selects promising candidates, the oracle
evaluates them, and the newly labelled candidates are added to the training set. This iterative process
continues for a specified number of acquisition rounds.

After all rounds complete, ``DesignTask`` computes ``auc_top_k`` — the normalised area under the
per-round top-k mean curve — and emits it as a ``campaign_summary`` log entry. This gives a single
sample-efficiency score for the full experiment. The metric uses ``top_k_mean`` for regression tasks
and ``accuracy`` for classification tasks, and is skipped silently when fewer than two rounds produce
a valid value.

.. automodule:: alf_core.tasks.design_task
   :members:
   :show-inheritance:
   :undoc-members:
