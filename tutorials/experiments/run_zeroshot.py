"""
Zero-shot learning experiment for GFP dataset.

This script demonstrates how to run a zero-shot task using:
- GFP dataset with sequence modality
- Random surrogate model
"""

from alf.tools.datasets.gfp import GFP
from alf.tools.models.random import RandomModel
from alf.core.surrogate.surrogate import Surrogate
from alf.core.utils.logger import TerminalLogger
from alf.core.tasks.zeroshot_task import ZeroShotTask


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
surrogate = Surrogate(model=RandomModel())


# Run zero-shot task
task = ZeroShotTask()
state = task.setup(dataset=gfp_dataset, surrogate=surrogate)
task.run(
    state,
    logger=TerminalLogger(),
)
