MLIP Model
==========

A machine-learned interatomic potential (MLIP) surrogate built around the
``mlip-jax`` MACE force field. Accepts ASE ``Atoms`` objects stored in ``Candidate.data``
and predicts per-structure energies, making it suitable for active learning over
atomistic systems.

By default the model finetunes from a pretrained foundation model; setting
``MLIPModelConfig.model_path=None`` trains a MACE network from scratch. When finetuning,
weights are reinitialised to the pretrained values at the start of every ``train()`` call,
so each active-learning iteration starts from the same foundation model rather than the
previous iteration's fit.

Key properties:

- **Input**: ASE ``Atoms`` objects stored in ``Candidate.data``
- **Output**: Per-structure energy means
- **Dynamic training**: With ``MLIPTrainConfig.dynamic_training=True``, ``batch_size``,
  ``learning_rate``, and ``epochs`` are auto-tuned from the training-set size to keep the
  total number of gradient updates roughly constant (~1000).
- **Weight-flip schedule**: With ``use_weight_flip=True``, the loss starts forces-weighted
  and switches to energy-weighted at ``flip_epoch``.
- **Reference energies**: Per-element reference energies (e0s) can be precomputed once and
  reused across iterations via the ``precomputed_e0s`` argument.
- **Sampling**: Not supported — raises ``NotImplementedError``.

.. note::

   This model requires the optional ``mlip`` dependency. Install it with the ``[mlip]`` extra:

   .. code-block:: bash

      pip install "alf-tools[mlip]"

   or from source:

   .. code-block:: bash

      pip install "alf_tools[mlip] @ git+https://github.com/instadeepai/alf.git#subdirectory=tools"

.. automodule:: alf_tools.models.mlip
   :members:
   :show-inheritance:
   :undoc-members:
