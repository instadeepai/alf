ESM2 Model
==========

A protein language model surrogate using `ESM-2 <https://huggingface.co/docs/transformers/model_doc/esm>`_
as the backbone. Accepts amino acid sequences as inputs and returns scalar fitness predictions.

The model fine-tunes a pre-trained ESM-2 encoder with a regression head. Embeddings can also be
extracted directly via ``featurise()`` for use with downstream models.

Key properties:

- **Input**: Amino acid sequences stored in ``Candidate.data``
- **Output**: Mean-only scalar predictions (no uncertainty estimates)
- **Backbone**: Configurable ESM-2 checkpoint (default: ``facebook/esm2_t6_8M_UR50D``)
- **Sampling**: Not supported — raises ``NotImplementedError``

.. note::

   This model requires the optional ``esm2`` dependency. Install it with the ``[esm2]`` extra:

   .. code-block:: bash

      pip install "alf-tools[esm2]"

   or from source:

   .. code-block:: bash

      pip install "git+https://github.com/instadeepai/alf.git#subdirectory=tools[esm2]"

.. automodule:: alf_tools.models.esm2
   :members:
   :show-inheritance:
   :undoc-members:
