ESM2 Model
==========

A protein language model surrogate using `ESM-2 <https://huggingface.co/docs/transformers/model_doc/esm>`_
as the backbone. Accepts amino acid sequences as inputs and supports two operating modes,
controlled by ``ESM2TrainConfig.linear_head``:

- **linear_head=True** (default): the ESM-2 backbone is frozen and a trainable linear head
  is stacked on top of pooled sequence embeddings. Use ``loss_fn='mse'`` for regression
  (``output_dim=1``) or ``loss_fn='cross_entropy'`` for classification (``output_dim=N``).
  Call ``train()`` to fit the head on labelled data. Sequence embeddings can also be
  extracted via ``embed()`` for use with downstream models.
- **linear_head=False**: no head is trained. ``predict()`` returns per-sequence
  pseudo-log-likelihood scores by masking all non-special tokens and computing the mean
  log-probability over those positions under the pre-trained model. This is a pure
  zero-shot scorer — ``train()`` raises ``NotImplementedError``.

The ESM-2 backbone is **always frozen**; full backbone fine-tuning is not currently supported.

Key properties:

- **Input**: Amino acid sequences stored in ``Candidate.data``
- **Output**: Mean-only scalar predictions (no uncertainty estimates)
- **Backbone**: Configurable ESM-2 checkpoint (default: ``facebook/esm2_t6_8M_UR50D``)
- **Sampling**: Not supported — raises ``NotImplementedError``
- **Tokenisation**: ``featurise()`` converts sequences to ``input_ids`` and
  ``attention_mask`` tensors for the ESM-2 tokeniser. It does **not** produce embeddings —
  use ``embed()`` for that.

.. note::

   This model requires the optional ``esm2`` dependency. Install it with the ``[esm2]`` extra:

   .. code-block:: bash

      pip install "alf_tools[esm2] @ git+https://github.com/instadeepai/alf.git#subdirectory=tools"

.. automodule:: alf_tools.models.esm2
   :members:
   :show-inheritance:
   :undoc-members:
