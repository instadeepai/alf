Matbench Dataset
=================

The `Matbench <https://matbench.materialsproject.org/>`_ benchmark dataset implementation.
Matbench provides 13 materials-property prediction tasks with predefined 5-fold
cross-validation splits, covering both composition-based (chemical formula) and
structure-based (crystal structure) inputs.

Both input kinds are stored under the ``TABULAR`` :term:`modality <Modality>` —
composition and structure values are `pymatgen <https://pymatgen.org/>`_
``Composition`` and ``Structure`` objects respectively, and both are serialised
identically to a JSON string (via their shared ``MSONable`` interface) before being
stored in ``Candidate.data``. Which pymatgen type a given task uses only affects how
that JSON string round-trips (``Composition`` vs. ``Structure``), not how ALF handles it.

**Supported tasks:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 14 10 20

   * - Task
     - Input type
     - Problem type
     - Samples
     - Target
   * - ``matbench_steels``
     - composition
     - regression
     - 312
     - yield strength (MPa)
   * - ``matbench_expt_gap``
     - composition
     - regression
     - 4,604
     - experimental gap (eV)
   * - ``matbench_expt_is_metal``
     - composition
     - classification
     - 4,921
     - is_metal
   * - ``matbench_glass``
     - composition
     - classification
     - 5,680
     - glass-forming ability
   * - ``matbench_dielectric``
     - structure
     - regression
     - 4,764
     - refractive index
   * - ``matbench_jdft2d``
     - structure
     - regression
     - 636
     - exfoliation energy
   * - ``matbench_log_gvrh``
     - structure
     - regression
     - 10,987
     - log10(shear modulus)
   * - ``matbench_log_kvrh``
     - structure
     - regression
     - 10,987
     - log10(bulk modulus)
   * - ``matbench_mp_e_form``
     - structure
     - regression
     - 132,752
     - formation energy
   * - ``matbench_mp_gap``
     - structure
     - regression
     - 106,113
     - band gap (eV)
   * - ``matbench_mp_is_metal``
     - structure
     - classification
     - 106,113
     - is_metal
   * - ``matbench_perovskites``
     - structure
     - regression
     - 18,928
     - formation energy
   * - ``matbench_phonons``
     - structure
     - regression
     - 1,265
     - last phonon DOS peak

``problem_type`` is derived automatically from the task — ``ProblemType.REGRESSION`` for
regression tasks, ``ProblemType.BINARY`` for the three classification tasks (all are
two-class) — and does not need to be set in ``MatbenchConfig``.

**Fold mode vs. merged mode:**

``MatbenchConfig.fold_number`` selects between two ways of using Matbench's predefined
5-fold cross-validation:

- **Fold mode** (``fold_number`` set to ``0``-``4``): uses Matbench's predefined
  train/test split for that fold directly. ``train_ratio`` controls what fraction of
  the Matbench train rows form the initial labelled training set (the remainder becomes
  ``candidate_pool``, capped at ``max_candidate_pool`` if set); ``validation_frac``
  carves a validation set out of that. ``test_ratio`` and ``split_type`` are **ignored**
  — the full Matbench test set for that fold is used as ``test``, since benchmark-
  comparable results require Matbench's exact predefined test rows.
- **Merged mode** (``fold_number=None``): all 5 folds are combined into one dataset and
  split using the standard ratio-based ``train_ratio``/``validation_frac``/
  ``test_ratio``/``split_type``. This loses Matbench's benchmark integrity guarantees —
  results are no longer directly comparable to published Matbench leaderboard scores.

In both modes, every candidate's ``features["fold_id"]`` records which of Matbench's 5
folds (``0``-``4``) it originally belonged to, for traceability.

**Dependencies:**

Requires the optional ``matbench`` extra (``pip install "alf-tools[matbench]"`` or
``alf_tools[materials]``), which installs ``matbench`` and ``pymatgen``. Data is
downloaded and cached automatically by the `matbench <https://pypi.org/project/matbench/>`_
package on first use.

.. automodule:: alf_tools.datasets.matbench
   :members:
   :show-inheritance:
   :undoc-members:
