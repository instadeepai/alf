# Tutorials

Learning-oriented, step-by-step guides that take you from zero to a working active-learning
experiment. The notebooks live on GitHub (with committed outputs); this page curates them into a
learning path.

## 🧪 Start with an experiment

Run the full ask/tell loop end to end:

- [Offline Design Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/experiments/offline_design_tutorial.ipynb): the loop driven by labels from a held-out, pre-scored pool. The best entry point.
- [Online Design Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/experiments/online_design_tutorial.ipynb): the same loop, with labels from a live scorer.

## 🪶 Lightweight (alf-core only)

If you want to explore ALF's core abstractions with minimal dependencies, start here:

- [ALF Core Quickstart](https://github.com/instadeepai/alf/blob/main/tutorials/alf_core_quickstart.ipynb): an active learning loop on MNIST digit classification, built entirely on `alf-core` using numpy and scipy. Implements a softmax-regression surrogate and an uncertainty-sampling acquisition function — both custom, neither available in `alf-tools` — to show how to bring your own components.

## 🤖 Go deeper on models

Each tutorial focuses on a specific surrogate model type—when to use it, how to configure it,
and what its uncertainty estimates look like.

- [GP Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/gp_tutorial.ipynb): Gaussian Process surrogate, and its [kernel cheat-sheet](https://github.com/instadeepai/alf/blob/main/tutorials/models/gp_kernel_cheatsheet.md). Good default for small datasets where uncertainty calibration matters.
- [CNN Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/cnn_tutorial.ipynb): convolutional sequence surrogate. Scales better than GPs for larger candidate pools.
- [Ensemble Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/ensemble_tutorial.ipynb): seed ensembles, MC dropout, and combined ensembles for uncertainty-aware prediction. Useful when you need calibrated uncertainty without a full Bayesian model.
- [ESM-2 Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/esm2_tutorial.ipynb): using a protein language model as a surrogate or zero-shot scorer. Useful when sequence context matters and you have limited labelled data.
- [Chemprop MPNN Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/chemprop_tutorial.ipynb): active learning for small molecules using SMILES inputs and the Chemprop message-passing neural network. The right choice when working with molecular graphs rather than sequences.
- [MLIP Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/mlip_design_tutorial.ipynb): a machine learning interatomic potential model (MACE force field) as a surrogate over atomistic systems, predicting per-structure energies from structure data. The right choice when designing over 3D molecular or crystal structures.

## 🗄️ Go deeper on datasets

- [GuacaMol Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/datasets/guacamol_tutorial.ipynb): exploring the GuacaMol drug-like molecule corpus—download, property analysis, and SMILES querying. A good starting point for small-molecule experiments.
- [ProteinGym Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/datasets/proteingym_tutorial.ipynb): the offline design loop applied to a ProteinGym deep-mutational-scanning (DMS) fitness benchmark. Best read after the Offline Design Tutorial, which it mirrors on a real protein-fitness dataset.

To add your own model, dataset, acquisition or search function, see the
[How-to / Recipes](../how-to/index.md).
