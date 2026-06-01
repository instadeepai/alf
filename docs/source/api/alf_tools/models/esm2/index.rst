ESM2 Model
==========

A protein language model surrogate using `ESM-2 <https://huggingface.co/docs/transformers/model_doc/esm>`_
as the backbone. Accepts amino acid sequences as inputs and supports two operating modes:

- **log_likelihood** (default): scores each sequence by its per-residue pseudo-log-likelihood
  under the pre-trained ESM-2 model. With ``freeze_backbone=True`` (the default), ``train()``
  is a no-op and the model acts as a pure zero-shot scorer. Set ``freeze_backbone=False`` to
  fine-tune the backbone with full masked language modelling.
- **mlp_head**: freezes the ESM-2 backbone and trains a linear head on top of pooled sequence
  embeddings. Supports regression (``output_dim=1``, ``mlp_loss='mse'``) and classification
  (``output_dim=N``, ``mlp_loss='cross_entropy'``). Sequence embeddings can also be extracted
  directly via ``embed()`` for use with downstream models.

Key properties:

- **Input**: Amino acid sequences stored in ``Candidate.data``
- **Output**: Mean-only scalar predictions (no uncertainty estimates)
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
