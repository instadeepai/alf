"""Factory functions: bridge Hydra DictConfig to existing ALF Python objects."""

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.model.base_model import BaseModel
from alf_core.oracle.oracle import Oracle
from alf_core.optimizer.optimizer import Optimizer
from alf_core.tasks.base_task import BaseTask
from alf_core.tasks.design_task import DesignTask
from alf_core.tasks.supervised_task import SupervisedTask
from alf_core.tasks.zeroshot_task import ZeroShotTask
from alf_core.utils.state_logger import FileStateLogger, StateLogger, TerminalStateLogger
from alf_tools.datasets.gfp import GFP
from alf_tools.datasets.proteingym import ProteinGym, ProteinGymConfig
from alf_tools.datasets.flip import FLIP, FLIPConfig


import sys

# Registry maps class_name -> (dataset_attr, config_attr) attribute names in this module.
# Resolved at call time so that test patches on module attributes take effect.
_DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "gfp": ("GFP", "BaseDatasetConfig"),
    "proteingym": ("ProteinGym", "ProteinGymConfig"),
    "flip": ("FLIP", "FLIPConfig"),
}


def build_dataset(cfg: DictConfig) -> BaseDataset:
    """Build a BaseDataset from the dataset config group.

    Args:
        cfg: Root Hydra DictConfig containing a `dataset` key.

    Returns:
        Constructed dataset (not yet split — call dataset.setup() separately).

    Raises:
        KeyError: If cfg.dataset.class_name is not registered.
    """
    dcfg = cfg.dataset
    cls_name, config_name = _DATASET_REGISTRY[dcfg.class_name]
    module = sys.modules[__name__]
    cls = getattr(module, cls_name)
    config_cls = getattr(module, config_name)
    fields = {k: v for k, v in OmegaConf.to_container(dcfg, resolve=True).items()
              if k != "class_name"}
    if "modality" in fields and isinstance(fields["modality"], str):
        fields["modality"] = fields["modality"].lower()
    dataset_config = config_cls(**fields)
    return cls(dataset_config)
