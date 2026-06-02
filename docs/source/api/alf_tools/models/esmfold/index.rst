ESMFold Model
=============

An inference-only oracle that runs `ESMFold <https://huggingface.co/facebook/esmfold_v1>`_ structure
prediction on amino acid sequences and returns a scalar confidence score per candidate. Suitable for
use as a black-box oracle inside ``Oracle(scorer=ESMFoldModel(ESMFoldModelConfig(...)))``.

Three scoring metrics are available, all in the range **[0, 1]** (higher is better):

.. list-table:: Scoring metric interpretation
   :header-rows: 1
   :widths: 15 45 40

   * - Metric
     - What it measures
     - Interpretation thresholds
   * - ``ptm`` *(default)*
     - Global structural plausibility of the entire fold. Analogous to TM-score against a
       hypothetical template; captures whether the sequence adopts a coherent 3-D topology.
     - | > 0.5 — confident fold, well-defined topology
       | 0.2–0.5 — moderate confidence, partially structured
       | < 0.1 — low confidence; typical for disordered proteins or peptides < ~20 residues
   * - ``mean_plddt``
     - Per-residue predicted local distance difference test (pLDDT) score, averaged over all
       non-padding residues. Measures local structural accuracy at the residue level.
       (ESMFold normalises pLDDT to [0, 1] internally.)
     - | > 0.7 — well-structured, confident local geometry
       | 0.5–0.7 — moderate confidence, flexible or partially ordered regions
       | < 0.5 — low confidence, likely disordered or unreliable residues
   * - ``combined``
     - Weighted average ``w * ptm + (1 − w) * mean_plddt`` (default ``w = 0.5``). Balances
       global topology confidence with local residue accuracy.
     - Inherits the [0, 1] range; apply the same per-metric thresholds above to each component
       before interpreting the combined score.

Key properties:

- **Input**: Amino acid sequences stored in ``Candidate.data`` (``Modality.SEQUENCE``)
- **Output**: Mean-only scalar predictions (no uncertainty estimates)
- **Batching**: ``batch_size > 1`` is supported only for ``scoring_metric='mean_plddt'``; pTM/combined require ``batch_size=1``
- **CPU use**: Backbone is automatically cast to fp32 on CPU to avoid fp16 numerical errors
- **Memory**: Pass ``chunk_size`` to reduce peak memory for long sequences; use ``cleanup()`` to free GPU memory after inference

.. note::

   This model requires the optional ``esmfold`` dependency. Install it with:

   .. code-block:: bash

      pip install "transformers>=4.36.0" "accelerate>=0.26.0"

.. automodule:: alf_tools.models.esmfold
   :members:
   :show-inheritance:
   :undoc-members:
