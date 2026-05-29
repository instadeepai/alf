.. ALF documentation master file:

ALF
=========

Overview
--------

**ALF** is an active learning framework for iterative optimization of biological and chemical design targets. It provides modular components for candidate search, surrogate modelling, and acquisition — enabling efficient discovery across expensive, high-dimensional search spaces such as protein sequences and small molecules.

.. note::

   This project is under active development.

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.12 or higher
- `uv <https://docs.astral.sh/uv/>`_ (recommended) or ``pip``

Installation
~~~~~~~~~~~~

**Quick Install**

Install ALF packages directly from GitHub:

.. code-block:: bash

   # Install the core package (minimal dependencies, no PyTorch required)
   pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

   # Install the tools package (includes PyTorch, models, and datasets)
   pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools

**Authentication:** Add your GitHub credentials to ``~/.netrc``:

.. code-block:: text

   machine github.com login <USERNAME> password <TOKEN>

**Development Install**

To develop ALF locally:

.. code-block:: bash

   git clone git@github.com:instadeepai/alf.git
   cd alf
   uv sync

For GPU support and full installation details, see the `Installation Guide <https://github.com/instadeepai/alf/blob/main/docs/INSTALLATION.md>`_.

Quick Start
~~~~~~~~~~~

Run an active learning design experiment:

.. code-block:: python

   from alf_core import (
       DatasetSearch,
       DesignTask,
       Optimizer,
       Oracle,
       Surrogate,
       TerminalTaskStateLogger,
   )
   from alf_tools.datasets.gfp import GFP
   from alf_tools.models.cnn import CNNModel
   from alf_tools.optimizer.acquisition_functions.greedy import Greedy

   # Set up dataset and components
   dataset = GFP(name="gfp", modality="sequence", seed=42)
   surrogate = Surrogate(model=CNNModel())
   optimizer = Optimizer(acquisition_fn=Greedy(), search_fn=DatasetSearch())
   oracle = Oracle(scorer=dataset)

   # Run the design task
   task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
   state = task.setup(dataset=dataset, surrogate=surrogate)
   task.run(
       state=state,
       task_state_loggers=[TerminalTaskStateLogger()],
       optimizer=optimizer,
       oracle=oracle,
   )

Tutorials
~~~~~~~~~

- `Offline Design Tutorial <https://github.com/instadeepai/alf/blob/main/tutorials/offline_design_tutorial.ipynb>`_ — Step-by-step guide to offline design experiments
- `Online Design Tutorial <https://github.com/instadeepai/alf/blob/main/tutorials/online_design_tutorial.ipynb>`_ — Guide to online design experiments
- `Ensemble Tutorial <https://github.com/instadeepai/alf/blob/main/tutorials/models/ensemble_tutorial.ipynb>`_ — Seed ensembles, MC dropout, and combined ensembles for uncertainty-aware prediction


Contents
--------

.. toctree::
   :maxdepth: 2

   Core <api/alf_core/index>
   Tools <api/alf_tools/index>
