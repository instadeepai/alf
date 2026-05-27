MPNN Model (Chemprop)
====================

A Message Passing Neural Network (MPNN) surrogate model using `Chemprop v2.x <https://chemprop.readthedocs.io/>`_
as the backbone. Accepts SMILES strings as inputs and returns scalar fitness predictions.

The model uses bond-based message passing (``BondMessagePassing``), a configurable graph-level
aggregation function, and a feed-forward regression head (``RegressionFFN``). The network is
initialised lazily on the first ``train()`` call; subsequent calls fine-tune from the current
weights rather than resetting them.

Key properties:

- **Input**: SMILES strings stored in ``Candidate.data``
- **Output**: Mean-only scalar predictions (no uncertainty estimates)
- **Aggregation**: ``mean`` (default), ``sum``, or ``norm``
- **Sampling**: Not supported — raises ``NotImplementedError``

.. note::

   This model requires the ``chemprop>=2.0.0`` package. Install the tools package
   with ``pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools``.

.. automodule:: alf_tools.models.chemprop
   :members:
   :show-inheritance:
   :undoc-members:
