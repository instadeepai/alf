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
from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig, SubsampleConfig
from alf_tools.models.gp import GPModel, GPModelConfig, GPTrainConfig, FeaturizerConfig
from alf_tools.models.mlp import MLPModel, MLPModelConfig, MLPTrainConfig
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig


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


def build_model(cfg: DictConfig) -> BaseModel:
    """Build a BaseModel from the model config group.

    Args:
        cfg: Root Hydra DictConfig containing a `model` key.

    Returns:
        Constructed model instance.

    Raises:
        KeyError: If cfg.model.class_name is not a known model type.
    """
    mcfg = cfg.model
    dispatch: dict[str, object] = {
        "gp": _build_gp,
        "mlp": _build_mlp,
        "cnn": _build_cnn,
        "ensemble": _build_ensemble,
    }
    builder = dispatch[mcfg.class_name]
    return builder(mcfg)


def _build_gp(mcfg: DictConfig) -> GPModel:
    """Build a GPModel from a model sub-config.

    Args:
        mcfg: The model sub-config (cfg.model).

    Returns:
        Configured GPModel instance.
    """
    lengthscale_prior = instantiate(mcfg.lengthscale_prior) if mcfg.lengthscale_prior else None
    outputscale_prior = instantiate(mcfg.outputscale_prior) if mcfg.outputscale_prior else None
    noise_constraint = instantiate(mcfg.noise_constraint) if mcfg.noise_constraint else None
    model_config = GPModelConfig(
        kernel_type=mcfg.kernel_type,
        ard=mcfg.ard,
        matern_nu=mcfg.matern_nu,
        mean_type=mcfg.mean_type,
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=outputscale_prior,
        noise_constraint=noise_constraint,
    )
    train_fields = {k: v for k, v in OmegaConf.to_container(mcfg.train, resolve=True).items()}
    train_config = GPTrainConfig(**train_fields)
    featurizer_config = FeaturizerConfig(featurizer_type=mcfg.get("featurizer_type", "one_hot"))
    return GPModel(
        name=mcfg.name,
        model_config=model_config,
        train_config=train_config,
        featurizer_config=featurizer_config,
    )


def _build_mlp(mcfg: DictConfig) -> MLPModel:
    """Build an MLPModel from a model sub-config.

    Args:
        mcfg: The model sub-config (cfg.model or cfg.model.member for ensembles).

    Returns:
        Configured MLPModel instance.
    """
    model_fields = {
        k: v for k, v in OmegaConf.to_container(mcfg, resolve=True).items()
        if k not in ("class_name", "name", "train")
    }
    model_config = MLPModelConfig(**model_fields)
    train_fields = {k: v for k, v in OmegaConf.to_container(mcfg.train, resolve=True).items()}
    train_config = MLPTrainConfig(**train_fields)
    return MLPModel(name=mcfg.name, model_config=model_config, train_config=train_config)


def _build_cnn(mcfg: DictConfig) -> CNNModel:
    """Build a CNNModel from a model sub-config.

    Args:
        mcfg: The model sub-config (cfg.model).

    Returns:
        Configured CNNModel instance.
    """
    model_fields = {
        k: v for k, v in OmegaConf.to_container(mcfg, resolve=True).items()
        if k not in ("class_name", "name", "train")
    }
    model_config = CNNModelConfig(**model_fields)
    train_fields = {k: v for k, v in OmegaConf.to_container(mcfg.train, resolve=True).items()}
    train_config = CNNTrainConfig(**train_fields)
    return CNNModel(name=mcfg.name, model_config=model_config, train_config=train_config)


def _build_ensemble(mcfg: DictConfig) -> EnsembleWrapper:
    """Build an EnsembleWrapper from a model sub-config.

    Args:
        mcfg: The model sub-config (cfg.model), must have class_name='ensemble'.

    Returns:
        EnsembleWrapper with n_members members, each built by model_factory.
    """
    subsample = None
    if mcfg.get("subsample") is not None:
        s = mcfg.subsample
        subsample = SubsampleConfig(fraction=s.fraction, replace=s.replace)
    ensemble_cfg = EnsembleWrapperConfig(
        base_seed=mcfg.base_seed,
        n_members=mcfg.n_members,
        subsample=subsample,
    )

    def model_factory(seed: int) -> BaseModel:
        member_cfg = OmegaConf.merge(mcfg.member, {"model_seed": seed})
        dispatch = {"mlp": _build_mlp, "cnn": _build_cnn, "gp": _build_gp}
        return dispatch[member_cfg.class_name](member_cfg)

    return EnsembleWrapper(
        model_factory=model_factory,
        config=ensemble_cfg,
        name=mcfg.name,
    )
