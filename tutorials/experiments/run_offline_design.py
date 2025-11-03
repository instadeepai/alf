"""
Offline design experiment for GFP dataset.

This script demonstrates how to run an offline design task using:
- GFP dataset with sequence modality
- CNN surrogate model
- Greedy acquisition function
- Dataset-based search strategy
- Multiple acquisition rounds with batch sampling
"""

from alf.tools.datasets.gfp import GFP
from alf.tools.models.cnn import CNNModel
from alf.core.surrogate.surrogate import Surrogate
from alf.core.optimizer.optimizer import Optimizer
from alf.core.optimizer.search import DatasetSearch
from alf.core.oracle.oracle import Oracle
from alf.core.utils.logger import TerminalLogger
from alf.core.tasks.design_task import DesignTask
from alf.tools.optimizer.acquisition_functions.greedy import Greedy

# Dataset configuration
split_config = {
    "split_ratio": {
        "train": 0.1,
        "validation": 0.1,
        "test": 0.1
    },
    "split_type": "random"
}

# Initialize dataset
gfp_dataset = GFP(
    name="gfp",
    modality="sequence",
    seed=51505,
    split_config=split_config
)

# Initialize surrogate model
surrogate = Surrogate(model=CNNModel())

# Initialize acquisition function
acquisition_fn = Greedy()

# Initialize search strategy
search_fn = DatasetSearch()

# Initialize optimizer
optimizer = Optimizer(acquisition_fn=acquisition_fn, search_fn=search_fn)

# Initialize oracle
oracle = Oracle(module=gfp_dataset)

# Run offline design task
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=gfp_dataset, surrogate=surrogate)
task.run(
    state,
    logger=TerminalLogger(),
    optimizer=optimizer,
    oracle=oracle,
)
