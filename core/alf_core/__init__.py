# This file makes alf_core a Python package
from alf_core.dataclasses import (
    Candidate,
    LabeledCandidates,
    Modality,
    Predictions,
    Results,
    TaskState,
)
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.model.base_model import BaseModel
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.optimizer import Optimizer
from alf_core.optimizer.search import (
    BaseSearch,
    DatasetSearch,
    GeneratorSearch,
    ModelProtocolSearch,
    ProtocolSearch,
)
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate
from alf_core.tasks.base_task import BaseTask
from alf_core.tasks.design_task import DesignTask
from alf_core.tasks.supervised_task import SupervisedTask
from alf_core.tasks.zeroshot_task import ZeroShotTask
from alf_core.utils.task_state_logger import (
    FileTaskStateLogger,
    TaskStateLogger,
    TerminalTaskStateLogger,
)
