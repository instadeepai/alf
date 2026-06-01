ESMFold Model
=============

An inference-only oracle that runs `ESMFold <https://huggingface.co/facebook/esmfold_v1>`_ structure
prediction on amino acid sequences and returns a scalar confidence score per candidate. Suitable for
use as a black-box oracle inside ``Oracle(scorer=ESMFoldModel(ESMFoldModelConfig(...)))``.

Three scoring metrics are available:

- **ptm** (default): predicted template modelling score — overall structural plausibility of the fold.
- **mean_plddt**: per-residue local confidence, averaged over all valid atoms.
- **combined**: weighted sum of pTM and mean pLDDT, controlled by ``combined_ptm_weight``.

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
