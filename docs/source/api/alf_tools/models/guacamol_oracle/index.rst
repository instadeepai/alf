GuacaMol Oracle
================

An online oracle that scores arbitrary SMILES via a GuacaMol composite/multi-property
optimisation (MPO) benchmark task (e.g. ``osimertinib_mpo``), computed fresh via RDKit on
every ``predict()`` call. For benchmark tasks, ``GuacaMol(BaseDataset).query()`` already
computes scores the same way — there's no corpus lookup for that mode either, every SMILES is
re-scored by the task function. What this class avoids is everything else that comes with
``GuacaMol(BaseDataset)``: constructing one downloads and RDKit-scores GuacaMol's entire
~1.6M-molecule corpus up front, even if the corpus is never otherwise used, and it also returns
labels as a ``BaseDataset``, which makes :py:class:`Oracle <alf_core.oracle.oracle.Oracle>`
classify it as *offline* by type (see :py:class:`Oracle <alf_core.oracle.oracle.Oracle>`'s own
"the oracle is the dataset" vs. "the oracle is a model" contract). ``GuacaMolOracle`` is a plain
:py:class:`BaseModel <alf_core.model.base_model.BaseModel>` — no corpus, near-instant to
construct, and it degrades gracefully on invalid SMILES (returns ``0.0``) rather than raising,
which matters when a search protocol occasionally proposes something malformed. See
:doc:`Switch offline to online </how-to/switch-offline-online>` for the online/offline Oracle
contract this implements.

Composite MPO scores are preferred over a single raw physicochemical property (e.g. LogP, QED)
as an online AL target: raw properties are smooth functions of 2D structure that a surrogate can
learn from very little data, which leaves little room for acquisition strategy to matter. MPO
scores combine several thresholded/Gaussian sub-terms via a geometric mean, giving a sharper
"few genuinely good candidates, mostly redundant" reward landscape.

.. automodule:: alf_tools.models.guacamol_oracle
   :members:
   :show-inheritance:
   :undoc-members:
