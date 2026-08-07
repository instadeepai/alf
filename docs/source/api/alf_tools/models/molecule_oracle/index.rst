Molecule Oracle Model
=======================

A generic online oracle for molecules: scores arbitrary SMILES with any user-supplied
``(smiles: str) -> float`` function, computed fresh on every ``predict()`` call rather than
looked up from a fixed, precomputed corpus. Has no RDKit or GuacaMol dependency itself —
whatever the scorer function needs is on the caller, so it works with a GuacaMol benchmark
scorer, a raw RDKit descriptor, a hand-written composite function, or an ML-based property
predictor alike.

:doc:`GuacaMol Oracle Model </api/alf_tools/models/guacamol_oracle/index>` is a thin preset built
on top of this class, fixing the scorer to one of GuacaMol's named benchmark tasks.

.. automodule:: alf_tools.models.molecule_oracle
   :members:
   :show-inheritance:
   :undoc-members:
