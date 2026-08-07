GuacaMol Oracle Model
======================

A thin preset of :doc:`Molecule Oracle Model </api/alf_tools/models/molecule_oracle/index>` that
fixes the scorer to one of GuacaMol's named composite/multi-property optimisation (MPO) benchmark
tasks (e.g. ``osimertinib_mpo``), computed fresh via RDKit on every ``predict()`` call. Unlike the
GuacaMol dataset's corpus-lookup ``query()``, this model has no fixed corpus or precomputed
labels, so it can be wrapped in :py:class:`Oracle <alf_core.oracle.oracle.Oracle>` and used to
score any candidate a search protocol proposes — including molecules that never appeared in any
corpus. See :doc:`Switch offline to online </how-to/switch-offline-online>` for the online/offline
Oracle contract this implements.

Composite MPO scores are preferred over a single raw physicochemical property (e.g. LogP, QED)
as an online AL target: raw properties are smooth functions of 2D structure that a surrogate can
learn from very little data, which leaves little room for acquisition strategy to matter. MPO
scores combine several thresholded/Gaussian sub-terms via a geometric mean, giving a sharper
"few genuinely good candidates, mostly redundant" reward landscape.

.. automodule:: alf_tools.models.guacamol_oracle
   :members:
   :show-inheritance:
   :undoc-members:
