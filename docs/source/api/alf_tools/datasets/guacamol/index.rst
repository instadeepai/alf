GuacaMol Dataset
================

The GuacaMol (Brown et al., 2019) benchmark dataset implementation. GuacaMol provides a
large corpus of ~1.6 million drug-like SMILES derived from ChEMBL, along with a suite of
goal-directed molecular design benchmarks. This implementation supports two complementary
modes: **property prediction** (RDKit physicochemical descriptors as regression targets) and
**benchmark tasks** (goal-directed scoring functions that evaluate molecular design quality).

Data is downloaded automatically from `Figshare <https://figshare.com/projects/GuacaMol/62224>`_
on first use and cached locally at ``~/.cache/alf/``.

**Supported physicochemical properties:**

+------------------------+------------------------------------------------------+
| Property               | Description                                          |
+========================+======================================================+
| ``BertzCT``            | Bertz complexity index                               |
+------------------------+------------------------------------------------------+
| ``MolLogP``            | Wildman–Crippen partition coefficient (lipophilicity)|
+------------------------+------------------------------------------------------+
| ``MolWt``              | Molecular weight (Da)                                |
+------------------------+------------------------------------------------------+
| ``TPSA``               | Topological polar surface area (Å²)                  |
+------------------------+------------------------------------------------------+
| ``NumHAcceptors``      | Number of hydrogen bond acceptors                    |
+------------------------+------------------------------------------------------+
| ``NumHDonors``         | Number of hydrogen bond donors                       |
+------------------------+------------------------------------------------------+
| ``NumRotatableBonds``  | Number of rotatable bonds                            |
+------------------------+------------------------------------------------------+
| ``NumAliphaticRings``  | Number of aliphatic rings                            |
+------------------------+------------------------------------------------------+
| ``NumAromaticRings``   | Number of aromatic rings                             |
+------------------------+------------------------------------------------------+
| ``QED``                | Quantitative Estimate of Drug-likeness               |
+------------------------+------------------------------------------------------+

**Supported benchmark tasks:**

+-------------------------------------+---------------------------------------------+
| Task                                | Category                                    |
+=====================================+=============================================+
| ``celecoxib_rediscovery``           | Rediscovery (ECFP4 Tanimoto)                |
+-------------------------------------+---------------------------------------------+
| ``troglitazone_rediscovery``        | Rediscovery (ECFP4 Tanimoto)                |
+-------------------------------------+---------------------------------------------+
| ``thiothixene_rediscovery``         | Rediscovery (ECFP4 Tanimoto)                |
+-------------------------------------+---------------------------------------------+
| ``aripiprazole_similarity``         | Similarity (FCFP4 Tanimoto, clipped ≤0.75) |
+-------------------------------------+---------------------------------------------+
| ``albuterol_similarity``            | Similarity (FCFP4 Tanimoto, clipped ≤0.75) |
+-------------------------------------+---------------------------------------------+
| ``mestranol_similarity``            | Similarity (atom-pair, clipped ≤0.75)       |
+-------------------------------------+---------------------------------------------+
| ``camphor_menthol_median``          | Median molecule (ECFP4)                     |
+-------------------------------------+---------------------------------------------+
| ``tadalafil_sildenafil_median``     | Median molecule (ECFP6)                     |
+-------------------------------------+---------------------------------------------+
| ``fexofenadine_mpo``                | Multi-property optimisation (MPO)           |
+-------------------------------------+---------------------------------------------+
| ``osimertinib_mpo``                 | Multi-property optimisation (MPO)           |
+-------------------------------------+---------------------------------------------+
| ``ranolazine_mpo``                  | Multi-property optimisation (MPO)           |
+-------------------------------------+---------------------------------------------+
| ``perindopril_mpo``                 | Multi-property optimisation (MPO)           |
+-------------------------------------+---------------------------------------------+
| ``amlodipine_mpo``                  | Multi-property optimisation (MPO)           |
+-------------------------------------+---------------------------------------------+
| ``sitagliptin_mpo``                 | Multi-property optimisation (MPO)           |
+-------------------------------------+---------------------------------------------+
| ``zaleplon_mpo``                    | Multi-property optimisation (MPO)           |
+-------------------------------------+---------------------------------------------+
| ``c7h8n2o2_isomer``                 | Isomer generation (C7H8N2O2)                |
+-------------------------------------+---------------------------------------------+
| ``c9h10n2o2pf2cl_isomer``           | Isomer generation (C9H10N2O2PF2Cl)          |
+-------------------------------------+---------------------------------------------+
| ``aripiprazole_scaffold_hop``       | Scaffold hopping (PHCO + SMARTS gates)      |
+-------------------------------------+---------------------------------------------+
| ``aripiprazole_decorator_hop``      | Decorator hopping (PHCO + SMARTS gates)     |
+-------------------------------------+---------------------------------------------+

Benchmark task scores are computed on-the-fly via RDKit — no pre-labelled corpus is required.
All scores are in [0, 1], aggregated using geometric or arithmetic means of sub-component scores
(Tanimoto similarity, Gaussian property penalties, SMARTS presence/absence checks).

**Split modes:**

+------------------+---------------------------------------------------------------+
| ``split_mode``   | Behaviour                                                     |
+==================+===============================================================+
| ``"random"``     | Random train/validation/test/candidate_pool split of the      |
|                  | combined corpus file (``guacamol_v1_all.smiles``).            |
+------------------+---------------------------------------------------------------+
| ``"low_vs_high"``| Split by target property value: low-value molecules form the  |
|                  | train set; high-value molecules form the test set.            |
+------------------+---------------------------------------------------------------+
| ``"stratified"`` | Stratified split on target property quantiles.                |
+------------------+---------------------------------------------------------------+
| ``"paper"``      | Uses the original train/valid/test file boundaries from       |
|                  | Brown et al. (2019), enabling direct comparison with published|
|                  | results. ``candidate_pool`` is empty in this mode.            |
+------------------+---------------------------------------------------------------+

**Split mapping to ALF (non-paper modes):**

+-----------------------------+-------------------------------------------------------+
| ALF split                   | Source                                                |
+=============================+=======================================================+
| ``train``                   | ``train_ratio`` fraction of the combined corpus       |
+-----------------------------+-------------------------------------------------------+
| ``validation``              | ``validation_frac`` of the train fraction             |
+-----------------------------+-------------------------------------------------------+
| ``test``                    | ``test_ratio`` fraction of the combined corpus        |
+-----------------------------+-------------------------------------------------------+
| ``candidate_pool``          | Remaining corpus after train + validation             |
|                             | (capped at ``max_candidate_pool`` if set)             |
+-----------------------------+-------------------------------------------------------+

**Paper split sizes:**

+--------------------+-----------+
| Split              | Molecules |
+====================+===========+
| Train              | 1,273,104 |
+--------------------+-----------+
| Validation         | 55,804    |
+--------------------+-----------+
| Test               | 55,821    |
+--------------------+-----------+
| **Total**          | 1,384,729 |
+--------------------+-----------+

**Property storage:**

Computed property values are stored in a single contiguous NumPy array on the dataset instance
rather than in individual ``Candidate.features`` dicts. After loading, ``GuacaMol._prop_matrix``
has shape ``(N, P)`` where *N* is the number of valid corpus molecules and *P* is the number of
requested properties; ``GuacaMol._prop_cols`` lists the property names in the corresponding column
order. ``Candidate.features`` is always ``{}`` for corpus molecules — accessing it will return an
empty dict, not a ``KeyError``.

Novel molecules queried via :meth:`~alf_tools.datasets.guacamol.GuacaMol.query` still receive a
populated ``features`` dict in the returned candidate (properties are computed on-the-fly by
RDKit), because they have no entry in ``_prop_matrix``.

Benchmark-task datasets do not compute RDKit properties on load; ``_prop_matrix`` retains its
empty sentinel shape of ``(0, 0)`` and ``_prop_cols`` is ``[]``.

**Novel molecules:** For property targets, SMILES not present in the loaded corpus are labelled
on-the-fly via RDKit in :meth:`~alf_tools.datasets.guacamol.GuacaMol.query`. This means
generative models can propose entirely new molecules and receive valid labels without reloading
the dataset.

.. automodule:: alf_tools.datasets.guacamol.guacamol_dataset
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: alf_tools.datasets.guacamol.guacamol_utils
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: alf_tools.datasets.guacamol.guacamol_scoring
   :members:
   :show-inheritance:
   :undoc-members:
