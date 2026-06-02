ESMFold Model
=============

A protein structure prediction oracle backed by Meta's ESMFold (``EsmForProteinFolding``
from HuggingFace). Unlike trainable models, ``ESMFoldModel`` is inference-only: it
does not implement ``train()``, ``featurise()``, or ``sample()``, and is intended for
use as an ``Oracle`` scorer rather than a ``Surrogate`` model.

Scores are reported as pTM, mean pLDDT, or a weighted combination of both.
``batch_size > 1`` is only valid with ``scoring_metric="mean_plddt"`` because ESMFold
returns a single pTM scalar per batch rather than one per sequence. Use ``chunk_size``
in ``ESMFoldModelConfig`` to reduce peak GPU memory on long sequences.

Requires ``transformers>=4.36.0`` and ``accelerate>=0.26.0``.

.. automodule:: alf_tools.models.esmfold
   :members:
   :show-inheritance:
   :undoc-members:
