FLIP Dataset
============

The FLIP (Fitness Landscape Inference for Proteins) benchmark dataset implementation. FLIP provides
standardised train/test splits across multiple protein fitness landscapes, designed to evaluate how
well models capture key aspects of protein sequence–function relationships.

Each dataset uses pre-defined splits (determined by the ``set`` column in the source CSV). Sequences
with ``set == "train"`` form the queryable pool; sequences with ``set == "test"`` are held out for
evaluation. FLIP's built-in ``validation`` flag is stored as a candidate feature but is **not** used
for splitting — ratio-based config parameters control all partitioning.

**Supported datasets and active splits:**

+---------------------+-----------------------------------------------------------+
| Dataset             | Active splits                                             |
+=====================+===========================================================+
| ``aav``             | ``des_mut``, ``mut_des``, ``one_vs_many``,                |
|                     | ``two_vs_many``, ``seven_vs_many``, ``low_vs_high``       |
+---------------------+-----------------------------------------------------------+
| ``gb1``             | ``one_vs_rest``, ``two_vs_rest``,                         |
|                     | ``three_vs_rest``, ``low_vs_high``                        |
+---------------------+-----------------------------------------------------------+
| ``meltome``         | ``mixed``, ``human``, ``human_cell``                      |
+---------------------+-----------------------------------------------------------+
| ``scl``             | ``mixed_soft``, ``mixed_hard``, ``human_soft``,           |
|                     | ``human_hard``, ``balanced``, ``mixed_vs_human_2``        |
+---------------------+-----------------------------------------------------------+
| ``sav``             | ``mixed``, ``human``, ``only_savs``                       |
+---------------------+-----------------------------------------------------------+

**Split mapping to ALF:**

+-----------------------------+----------------------------------------------------------+
| ALF split                   | FLIP source                                              |
+=============================+==========================================================+
| ``train``                   | ``train_ratio × FLIP train`` (minus validation fraction) |
+-----------------------------+----------------------------------------------------------+
| ``validation``              | ``validation_frac × (train_ratio × FLIP train)``         |
+-----------------------------+----------------------------------------------------------+
| ``test``                    | ``test_ratio × FLIP test``                               |
+-----------------------------+----------------------------------------------------------+
| ``candidate_pool``          | Remaining FLIP train after train + validation            |
|                             | (capped at ``max_candidate_pool`` if set)                |
+-----------------------------+----------------------------------------------------------+

The FLIP train set acts as the queryable pool for active learning. Sequences not allocated to
train or validation form the candidate pool, from which the model acquires new labels each round.
The test set is held out purely for evaluation.

Data is downloaded automatically from the `FLIP GitHub repository
<https://github.com/J-SNACKKB/FLIP>`_ on first use and cached locally at
``alf_tools/datasets/data/FLIP/{dataset}/splits.zip``.

**Dataset and split sizes:**

*AAV*

+--------------------+---------+---------+---------+
| Split              | Train   | Test    | Total   |
+====================+=========+=========+=========+
| ``des_mut``        | 201,426 | 82,583  | 284,009 |
+--------------------+---------+---------+---------+
| ``mut_des``        | 82,583  | 201,426 | 284,009 |
+--------------------+---------+---------+---------+
| ``one_vs_many``    | 1,170   | 81,413  | 82,583  |
+--------------------+---------+---------+---------+
| ``two_vs_many``    | 31,807  | 50,776  | 82,583  |
+--------------------+---------+---------+---------+
| ``seven_vs_many``  | 70,002  | 12,581  | 82,583  |
+--------------------+---------+---------+---------+
| ``low_vs_high``    | 47,546  | 35,037  | 82,583  |
+--------------------+---------+---------+---------+

*GB1*

+--------------------+-------+-------+-------+
| Split              | Train | Test  | Total |
+====================+=======+=======+=======+
| ``one_vs_rest``    | 28    | 8,705 | 8,733 |
+--------------------+-------+-------+-------+
| ``two_vs_rest``    | 424   | 8,309 | 8,733 |
+--------------------+-------+-------+-------+
| ``three_vs_rest``  | 2,990 | 5,743 | 8,733 |
+--------------------+-------+-------+-------+
| ``low_vs_high``    | 5,089 | 3,644 | 8,733 |
+--------------------+-------+-------+-------+

*Meltome*

+--------------------+--------+-------+--------+
| Split              | Train  | Test  | Total  |
+====================+========+=======+========+
| ``mixed``          | 24,817 | 3,134 | 27,951 |
+--------------------+--------+-------+--------+
| ``human``          | 8,148  | 1,945 | 10,093 |
+--------------------+--------+-------+--------+
| ``human_cell``     | 5,792  | 1,366 | 7,158  |
+--------------------+--------+-------+--------+

*SCL*

+----------------------+--------+-------+--------+
| Split                | Train  | Test  | Total  |
+======================+========+=======+========+
| ``mixed_soft``       | 11,181 | 2,768 | 13,949 |
+----------------------+--------+-------+--------+
| ``mixed_hard``       | 11,181 | 490   | 11,671 |
+----------------------+--------+-------+--------+
| ``human_soft``       | 11,181 | 577   | 11,758 |
+----------------------+--------+-------+--------+
| ``human_hard``       | 11,181 | 119   | 11,300 |
+----------------------+--------+-------+--------+
| ``balanced``         | 11,181 | 385   | 11,566 |
+----------------------+--------+-------+--------+
| ``mixed_vs_human_2`` | 28,303 | 1,717 | 30,020 |
+----------------------+--------+-------+--------+

*SAV*

+--------------------+---------+-------+---------+
| Split              | Train   | Test  | Total   |
+====================+=========+=======+=========+
| ``mixed``          | 103,174 | 6,275 | 109,449 |
+--------------------+---------+-------+---------+
| ``human``          | 14,425  | 651   | 15,076  |
+--------------------+---------+-------+---------+
| ``only_savs``      | 93,593  | 6,275 | 99,868  |
+--------------------+---------+-------+---------+

**Split types explained:**

*Mutation-order splits* test a model's ability to generalise from low-order mutants to higher-order
ones:

- ``one_vs_many`` / ``two_vs_many`` / ``seven_vs_many`` — train on sequences with exactly 1, 2, or
  7 mutations; test on sequences with more mutations. Used in ``aav``.
- ``one_vs_rest`` / ``two_vs_rest`` / ``three_vs_rest`` — train on a specific subset of variants
  at defined positions; test on all remaining variants. Used in ``gb1``.

*Fitness-range splits* test extrapolation beyond the training fitness range:

- ``low_vs_high`` — train on low-fitness sequences, test on high-fitness ones. Used in ``aav``
  and ``gb1``.

*Design vs. mutation splits* reflect different sequence generation strategies:

- ``des_mut`` — train on designed sequences, test on mutant sequences. Used in ``aav``.
- ``mut_des`` — the reverse: train on mutants, test on designed sequences. Used in ``aav``.

*Source/organism splits* test cross-organism or cross-context generalisation:

- ``mixed`` — sequences pooled across multiple organisms for training.
- ``human`` — training restricted to human-derived sequences.
- ``human_cell`` — training restricted to human cell-line sequences. Used in ``meltome``.
- ``only_savs`` — training restricted to single amino acid variants. Used in ``sav``.

*Label-threshold splits* (``scl`` only) vary the stringency of the positive class:

- ``mixed_soft`` / ``human_soft`` — lenient threshold for positive-class assignment.
- ``mixed_hard`` / ``human_hard`` — strict threshold; fewer positives in the test set.
- ``balanced`` — class-balanced test set.
- ``mixed_vs_human_2`` — train on mixed-organism data, test on human sequences with a stricter
  label threshold.

.. automodule:: alf_tools.datasets.flip
   :members:
   :show-inheritance:
   :undoc-members:
