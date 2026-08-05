ESM2 Model
==========

A protein language model surrogate using `ESM-2 <https://huggingface.co/docs/transformers/model_doc/esm>`_
as the backbone. Accepts amino acid sequences as inputs and supports three operating modes,
selected via ``ESM2TrainConfig.mode`` together with ``freeze_backbone``:

- **mode='linear_head'** (default): a trainable linear head is stacked on top of pooled
  sequence embeddings. With ``freeze_backbone=True`` (default) only the head is trained on
  top of the frozen backbone. With ``freeze_backbone=False`` the backbone is fine-tuned
  jointly with the head, using ``backbone_learning_rate`` (typically lower than
  ``learning_rate``) for the backbone parameters — usually slower to train but often more
  accurate. The head is configured via ``loss_fn`` and ``output_dim``:

  - ``loss_fn='mse'``: mean-squared-error regression. Set ``output_dim=1``.
    ``predict()`` returns raw scalar values.
  - ``loss_fn='cross_entropy'``: multi-class cross-entropy classification. Set ``output_dim=N``
    for N classes. Labels must be integers in ``[0, N)``; float labels are truncated to int
    with a warning. ``predict()`` returns the argmax class index as a float.

  Call ``train()`` to fit the head on labelled data (``loss_fn`` must be set). Sequence
  embeddings can also be extracted via :meth:`~alf_tools.models.esm2.ESM2Model.embed` for use with downstream models.
- **mode='likelihoods'** with ``freeze_backbone=True``: zero-shot scoring. ``predict()``
  returns per-sequence pseudo-log-likelihood (PLL) scores — each non-special token is masked
  one at a time and the log-probability of the correct residue at that position is accumulated,
  and the final score is the mean log-probability across all non-special positions. Nothing is
  trained, so ``train()`` raises ``NotImplementedError`` and ``loss_fn`` must be left ``None``.
- **mode='likelihoods'** with ``freeze_backbone=False``: MLM fine-tuning. ``train()``
  fine-tunes the **full ESM-2 backbone** via masked language modelling on the input sequences;
  ``loss_fn`` must be ``'mlm'``. Masking is controlled by ``mask_probability`` (fraction of
  eligible tokens masked per sequence) and ``mask_splitting`` (the ``(p_mask, p_random,
  p_unchanged)`` 3-way replacement probabilities, which must sum to 1.0). Labels in
  ``train_data`` are ignored, but :class:`~alf_core.dataclasses.labelled_candidates.LabelledCandidates` still requires them — pass placeholder
  values. After fine-tuning, ``predict()`` returns updated PLL scores. Training reports
  ``perplexity`` and ``token_accuracy`` over masked positions; validation masking is seeded
  from ``ESM2ModelConfig.seed`` so metrics are comparable across epochs.

Key properties:

- **Input**: Amino acid sequences stored in ``Candidate.data``
- **Output**: Mean-only scalar predictions (no uncertainty estimates)
- **Backbone**: Configurable ESM-2 checkpoint — specify via ``ESM2ModelConfig(model_id=...)``, e.g. ``facebook/esm2_t6_8M_UR50D``
- **Sampling**: Not supported — raises ``NotImplementedError``
- **Tokenisation**: ``featurise()`` converts sequences to ``input_ids`` and
  ``attention_mask`` tensors for the ESM-2 tokeniser. It does **not** produce embeddings —
  use :meth:`~alf_tools.models.esm2.ESM2Model.embed` for that.

.. note::

   This model requires the optional ``esm2`` dependency. Install it with the ``[esm2]`` extra:

   .. code-block:: bash

      pip install "alf-tools[esm2]"

.. automodule:: alf_tools.models.esm2
   :members:
   :show-inheritance:
   :undoc-members:

Configuration
-------------

.. automodule:: alf_tools.models.esm_utils.config
   :members:
   :show-inheritance:
   :undoc-members:

Loss and scoring
----------------

.. automodule:: alf_tools.models.esm_utils.loss
   :members:
   :undoc-members:

.. automodule:: alf_tools.models.esm_utils.scoring_function
   :members:
   :undoc-members:
