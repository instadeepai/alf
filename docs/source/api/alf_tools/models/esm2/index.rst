ESM2 Model
==========

A protein language model surrogate using `ESM-2 <https://huggingface.co/docs/transformers/model_doc/esm>`_
as the backbone. Accepts amino acid sequences as inputs and supports two operating modes,
controlled by ``ESM2TrainConfig.scoring_function``:

- **scoring_function='linear_head'** (default): the ESM-2 backbone is frozen and a trainable linear head
  is stacked on top of pooled sequence embeddings. The head is configured via ``loss_fn`` and
  ``output_dim`` in ``ESM2TrainConfig``:

  - ``loss_fn='mse'`` *(default)*: mean-squared-error regression. Set ``output_dim=1``.
    ``predict()`` returns raw scalar values.
  - ``loss_fn='cross_entropy'``: multi-class cross-entropy classification. Set ``output_dim=N``
    for N classes. Labels must be integers in ``[0, N)``; float labels are truncated to int
    with a warning. ``predict()`` returns the argmax class index as a float.

  Call ``train()`` to fit the head on labelled data. Sequence embeddings can also be
  extracted via ``embed()`` for use with downstream models.
- **scoring_function=None**: no head is trained. ``predict()`` returns per-sequence
  pseudo-log-likelihood scores by masking all non-special tokens and computing the mean
  log-probability over those positions under the pre-trained model. This is a pure
  zero-shot scorer — ``train()`` raises ``NotImplementedError``.

The ESM-2 backbone is **always frozen**; full backbone fine-tuning is not currently supported.

Key properties:

- **Input**: Amino acid sequences stored in ``Candidate.data``
- **Output**: Mean-only scalar predictions (no uncertainty estimates)
- **Backbone**: Configurable ESM-2 checkpoint — specify via ``ESM2ModelConfig(model_id=...)``, e.g. ``facebook/esm2_t6_8M_UR50D``
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
