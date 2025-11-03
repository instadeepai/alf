"""
Supervised learning experiment for GFP dataset.

This script demonstrates how to run a supervised task using:
- GFP dataset with sequence modality
- CNN surrogate model
- Greedy acquisition function
- Dataset-based search strategy
"""

from alf.tools.datasets.gfp import GFP
from alf.tools.models.cnn import CNNModel
from alf.core.surrogate.surrogate import Surrogate
from alf.core.oracle.oracle import Oracle
from alf.core.utils.logger import TerminalLogger
from alf.core.tasks.supervised_task import SupervisedTask


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

# Initialize oracle
oracle = Oracle(module=gfp_dataset)

# Run supervised task
task = SupervisedTask()
state = task.setup(dataset=gfp_dataset, surrogate=surrogate)
task.run(
    state,
    logger=TerminalLogger(),
    oracle=oracle,
)
