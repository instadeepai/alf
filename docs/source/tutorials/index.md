# Tutorials

Learning-oriented, step-by-step guides that take you from zero to a working active-learning
experiment. The notebooks live on GitHub (with committed outputs); this page curates them into a
learning path.

## Start with an experiment

Run the full ask/tell loop end to end:

- [Offline Design Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/experiments/offline_design_tutorial.ipynb): the loop driven by labels from a held-out, pre-scored pool. The best entry point.
- [Online Design Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/experiments/online_design_tutorial.ipynb): the same loop, with labels from a live scorer.

## Go deeper on models

- [GP Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/gp_tutorial.ipynb): Gaussian Process surrogate, and its [kernel cheat-sheet](https://github.com/instadeepai/alf/blob/main/tutorials/models/gp_kernel_cheatsheet.md).
- [CNN Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/cnn_tutorial.ipynb): convolutional sequence surrogate.
- [Ensemble Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/ensemble_tutorial.ipynb): seed ensembles, MC dropout, and combined ensembles for uncertainty-aware prediction.
- [ESM-2 Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/esm2_tutorial.ipynb): protein language model as a surrogate or zero-shot scorer.
- [Chemprop MPNN Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/models/chemprop_tutorial.ipynb): active learning for small molecules using SMILES inputs and the Chemprop MPNN.

## Go deeper on datasets

- [GuacaMol Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/datasets/guacamol_tutorial.ipynb): drug-like molecule corpus: download, property analysis, and SMILES querying.

To add your own model, dataset, acquisition or search function, see the
[How-to / Recipes](../how-to/index.md).
