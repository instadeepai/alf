Acquisition Batch Metrics
=========================

Metrics that evaluate the quality and diversity of the candidate batch acquired in each round.
Unlike the regression and classification registries, these metrics operate on
:class:`~alf_core.dataclasses.candidate.Candidate` objects or
:class:`~alf_core.dataclasses.LabelledCandidates` rather than prediction arrays.

:func:`~alf_core.utils.metrics.acquisition_batch.intra_batch_diversity` measures how spread out the acquired batch is in design space,
using normalised Levenshtein distance for SEQUENCE candidates and cosine distance for
TABULAR candidates (MOLECULE is not yet supported). :func:`~alf_core.utils.metrics.acquisition_batch.compute_recall` and :func:`~alf_core.utils.metrics.acquisition_batch.compute_regret` measure
the quality of the acquired set relative to the full candidate pool.

.. automodule:: alf_core.utils.metrics.acquisition_batch
   :members:
   :show-inheritance:
   :undoc-members:
