Design Task
===========

The ``DesignTask`` runs a multi-round active learning optimization loop. In each round, the surrogate
model is trained on the current training data, the optimizer selects promising candidates, the oracle
evaluates them, and the newly labelled candidates are added to the training set. This iterative process
continues for a specified number of acquisition rounds.

After all rounds complete, ``DesignTask`` computes ``auc_top_k`` — the normalised area under the
per-round top-k mean curve — and emits it as a ``experiment_summary`` log entry. This gives a single
sample-efficiency score for the full experiment. For regression tasks the curve is the ``top_k_mean``
of all candidates acquired so far; for classification tasks it is the per-round test-set accuracy.
The summary is skipped silently when fewer than two rounds produce a valid value.

.. automodule:: alf_core.tasks.design_task
   :members:
   :show-inheritance:
   :undoc-members:
