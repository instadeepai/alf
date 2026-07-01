MLIP Model
==========

A machine-learned interatomic potential (MLIP) surrogate built around the
open-source `mlip <https://github.com/instadeepai/mlip>`_ package. Accepts
structure dictionaries stored in ``Candidate.data`` and predicts per-structure
energies, making it suitable for active learning over atomistic systems.

With ``MLIPModelConfig.model_path=None`` the model trains from scratch. Set
``model_path`` to a pretrained checkpoint to finetune. ``MLIPModelConfig.model_type``
selects the underlying ``mlip`` architecture (``"mace"``, ``"nequip"``,
``"visnet"``, or ``"esen"``), and ``network_config`` accepts the corresponding
``mlip`` network config object (for example ``Mace.Config(...)``). When finetuning,
weights are reinitialised to the pretrained values at the start of every ``train()``
call, so each active-learning iteration starts from the same foundation model rather
than the previous iteration's fit.

Key properties:

- **Input**: ``Candidate.data`` dictionaries compatible with
  ``mlip.data.ChemicalSystem`` keyword arguments, such as ``atomic_numbers`` and
  ``positions``. Force labels for training are supplied as
  ``Candidate.features["forces"]``.
- **Output**: Standard ``Predictions`` with per-structure energy means.
- **Optimizer**: Native ``mlip`` optimizer config objects are accepted via
  ``MLIPTrainConfig.optimizer_config``
- **Loss**: Native ``mlip`` loss objects are accepted via ``MLIPTrainConfig.loss``;
  the default is ``MSELoss``.
- **Architectures**: MACE, NequIP, ViSNet, and eSEN via ``MLIPModelConfig.model_type``.
- **Reference energies**: Scratch training computes per-element reference energies
  from the training split. Finetuning derives target-domain reference energies from
  the training split and merges them with the pretrained species table so checkpoint
  parameter shapes remain compatible while the absolute energy zero is retargeted.
- **Sampling**: Not supported — raises ``NotImplementedError``.

.. note::

   This model requires the optional ``mlip`` dependency. Install the CPU/default
   build with the ``[mlip]`` extra:

   .. code-block:: bash

      pip install "alf-tools[mlip]"

   To opt into the CUDA 13 build, use the ``[mlip-cuda13]`` extra instead:

   .. code-block:: bash

      pip install "alf-tools[mlip-cuda13]"

   or from source:

   .. code-block:: bash

      pip install "alf_tools[mlip] @ git+https://github.com/instadeepai/alf.git#subdirectory=tools"

.. automodule:: alf_tools.models.mlip
   :members:
   :show-inheritance:
   :undoc-members:
