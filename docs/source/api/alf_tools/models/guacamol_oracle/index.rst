GuacaMol Oracle
================

An online oracle that scores arbitrary SMILES via a GuacaMol composite/multi-property
optimisation (MPO) benchmark task (e.g. ``osimertinib_mpo``), computed fresh via RDKit on
every ``predict()`` call. For benchmark tasks, ``GuacaMol(BaseDataset).query()`` already
scores this way — there's no corpus lookup for that mode either; every SMILES is re-scored
by the task function. What ``GuacaMolOracle`` avoids is everything else bundled into
``GuacaMol(BaseDataset)``. Constructing one downloads and RDKit-scores GuacaMol's entire
~1.6M-molecule corpus up front, whether or not that corpus ever gets used. It also returns
labels as a ``BaseDataset``, which is why :py:class:`Oracle <alf_core.oracle.oracle.Oracle>`
classifies it as *offline* by type (see :py:class:`Oracle <alf_core.oracle.oracle.Oracle>`'s
own "the oracle is the dataset" vs. "the oracle is a model" contract). ``GuacaMolOracle`` is
a plain :py:class:`BaseModel <alf_core.model.base_model.BaseModel>` instead: no corpus,
near-instant to construct, and it degrades gracefully on invalid SMILES (returns ``0.0``)
rather than raising — useful when a search protocol proposes something malformed. See
:doc:`Switch offline to online </how-to/switch-offline-online>` for the online/offline
Oracle contract this implements.

MPO scores beat a single raw physicochemical property (e.g. LogP, QED) as an online AL
target because raw properties are smooth functions of 2D structure — a surrogate learns
them from very little data, leaving acquisition strategy nothing to do. MPO scores combine
several thresholded/Gaussian sub-terms via a geometric mean instead, producing a sharper
reward landscape: a few genuinely good candidates, mostly redundant.

.. automodule:: alf_tools.models.guacamol_oracle
   :members:
   :show-inheritance:
   :undoc-members: