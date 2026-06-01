ESM2 Model
==========

A protein language model surrogate using `ESM-2 <https://huggingface.co/docs/transformers/model_doc/esm>`_
as the backbone. Accepts amino acid sequences as inputs and returns scalar fitness predictions
as per-sequence pseudo-log-likelihood scores. Sequence embeddings can also be extracted directly
via ``embed()`` for use with downstream models.

The model optionally fine-tunes the ESM-2 backbone with masked language modelling (MLM) or
full log-likelihood masking. With ``freeze_backbone=True`` (the default), ``train()`` is a no-op
and the model operates as a pure zero-shot scorer.

Key properties:

- **Input**: Amino acid sequences stored in ``Candidate.data``
- **Output**: Mean-only scalar predictions (pseudo-log-likelihood scores; no uncertainty estimates)
- **Backbone**: Configurable ESM-2 checkpoint (default: ``facebook/esm2_t6_8M_UR50D``)
- **Sampling**: Not supported — raises ``NotImplementedError``

.. note::

   This model requires the optional ``esm2`` dependency. Install it with the ``[esm2]`` extra:

   .. code-block:: bash

      pip install "alf_tools[esm2] @ git+https://github.com/instadeepai/alf.git#subdirectory=tools"

.. automodule:: alf_tools.models.esm2
   :members:
   :show-inheritance:
   :undoc-members:
