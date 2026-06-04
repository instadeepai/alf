# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory functions: bridge Hydra DictConfig to existing ALF Python objects."""

import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig  # noqa: F401
from alf_core.model.base_model import BaseModel
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.optimizer import Optimizer
from alf_core.oracle.oracle import Oracle
from alf_core.tasks.base_task import BaseTask
from alf_core.tasks.design_task import DesignTask
from alf_core.tasks.supervised_task import SupervisedTask
from alf_core.tasks.zeroshot_task import ZeroShotTask
from alf_core.utils.state_logger import FileStateLogger, StateLogger, TerminalStateLogger
from alf_tools.datasets.flip import FLIP, FLIPConfig  # noqa: F401
from alf_tools.datasets.gfp import GFP  # noqa: F401
from alf_tools.datasets.proteingym import ProteinGym, ProteinGymConfig  # noqa: F401
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig
from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig, SubsampleConfig
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.models.mlp import MLPModel, MLPModelConfig, MLPTrainConfig
from alf_tools.optimizer.acquisition_functions.core_set import CoreSet
from alf_tools.optimizer.acquisition_functions.expected_improvement import ExpectedImprovement
from alf_tools.optimizer.acquisition_functions.greedy import Greedy
from alf_tools.optimizer.acquisition_functions.thompson_sampling import ThompsonSampling
from alf_tools.optimizer.acquisition_functions.ucb import UCB
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Registry maps class_name -> (dataset_attr, config_attr) attribute names in this module.
# Resolved at call time so that test patches on module attributes take effect.
_DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "gfp": ("GFP", "BaseDatasetConfig"),
    "proteingym": ("ProteinGym", "ProteinGymConfig"),
    "flip": ("FLIP", "FLIPConfig"),
}

_ACQ_FN_REGISTRY: dict[str, type[AcquisitionFunction]] = {
    "ucb": UCB,
    "ei": ExpectedImprovement,
    "greedy": Greedy,
    "thompson": ThompsonSampling,
    "core_set": CoreSet,
}


def build_dataset(cfg: DictConfig) -> BaseDataset:
    """Build a BaseDataset from the dataset config group.

    Args:
        cfg: Root Hydra DictConfig containing a `dataset` key.

    Returns:
        Constructed dataset (not yet split — call dataset.setup() separately).

    Raises:
        KeyError: If cfg.dataset.class_name is not registered.
        ValueError: If cfg.dataset.class_name is registered but has missing config fields.
    """
    dcfg = cfg.dataset
    if dcfg.class_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset.class_name: {dcfg.class_name!r}. Valid: {sorted(_DATASET_REGISTRY)}"
        )
    cls_name, config_name = _DATASET_REGISTRY[dcfg.class_name]
    module = sys.modules[__name__]
    cls = getattr(module, cls_name)
    config_cls = getattr(module, config_name)
    fields = {
        k: v for k, v in OmegaConf.to_container(dcfg, resolve=True).items() if k != "class_name"
    }
    if "modality" in fields and isinstance(fields["modality"], str):
        # YAML uses uppercase (e.g. "SEQUENCE"); Modality enum values are lowercase ("sequence").
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
        ValueError: If cfg.model.class_name is 'ensemble' but
            cfg.model.member.class_name is not a known model type.
    """
    mcfg = cfg.model
    dispatch: dict[str, Callable[[DictConfig], BaseModel]] = {
        "gp": _build_gp,
        "mlp": _build_mlp,
        "cnn": _build_cnn,
        "ensemble": _build_ensemble,
    }
    if mcfg.class_name not in dispatch:
        raise ValueError(
            f"Unknown model.class_name: {mcfg.class_name!r}. Valid: {sorted(dispatch)}"
        )
    return dispatch[mcfg.class_name](mcfg)


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
    raw = OmegaConf.to_container(mcfg, resolve=True)
    valid_fields = {f.name for f in dataclasses.fields(MLPModelConfig)}
    model_config = MLPModelConfig(**{k: v for k, v in raw.items() if k in valid_fields})
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
    raw = OmegaConf.to_container(mcfg, resolve=True)
    valid_fields = {f.name for f in dataclasses.fields(CNNModelConfig)}
    model_config = CNNModelConfig(**{k: v for k, v in raw.items() if k in valid_fields})
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
    member_dispatch: dict[str, Callable[[DictConfig], BaseModel]] = {
        "mlp": _build_mlp,
        "cnn": _build_cnn,
        "gp": _build_gp,
    }

    def model_factory(seed: int) -> BaseModel:
        member_cfg = OmegaConf.merge(mcfg.member, {"model_seed": seed})
        return member_dispatch[member_cfg.class_name](member_cfg)

    return EnsembleWrapper(
        model_factory=model_factory,
        config=ensemble_cfg,
        name=mcfg.name,
    )


def _build_acq_fn(acq_cfg: DictConfig) -> AcquisitionFunction:
    """Build an AcquisitionFunction from an acquisition_fn sub-config.

    Args:
        acq_cfg: The acquisition_fn sub-config (cfg.optimizer.acquisition_fn).
            Must contain a `name` key mapping to a registered acquisition function.
            All other keys are forwarded as constructor keyword arguments (e.g. `alpha` for UCB).

    Returns:
        Constructed AcquisitionFunction instance.

    Raises:
        ValueError: If acq_cfg.name is not a registered acquisition function name.
    """
    name = acq_cfg.name
    if name not in _ACQ_FN_REGISTRY:
        raise ValueError(
            f"Unknown acquisition_fn.name: {name!r}. Valid: {sorted(_ACQ_FN_REGISTRY)}"
        )
    cls = _ACQ_FN_REGISTRY[name]
    kwargs = {k: v for k, v in OmegaConf.to_container(acq_cfg, resolve=True).items() if k != "name"}
    return cls(**kwargs)


def build_optimizer(cfg: DictConfig) -> Optimizer:
    """Build an Optimizer from the optimizer config group.

    Args:
        cfg: Root Hydra DictConfig containing an `optimizer` key.

    Returns:
        Constructed Optimizer with acquisition function and search function.
    """
    acq_fn = _build_acq_fn(cfg.optimizer.acquisition_fn)
    search_fn = instantiate(cfg.optimizer.search_fn)
    return Optimizer(acquisition_fn=acq_fn, search_fn=search_fn)


def build_oracle(cfg: DictConfig, dataset: BaseDataset | None) -> Oracle:
    """Build an Oracle from the oracle config group.

    Args:
        cfg: Root Hydra DictConfig containing an `oracle` key.
        dataset: Already-constructed dataset, used when oracle.mode == 'dataset'.

    Returns:
        Oracle wrapping either the dataset (offline) or an instantiated scorer (online).

    Raises:
        ValueError: If mode is 'dataset' but dataset is None.
        ValueError: If mode is unknown.
    """
    mode = cfg.oracle.mode
    if mode == "dataset":
        if dataset is None:
            raise ValueError("oracle.mode='dataset' requires a dataset, got None")
        return Oracle(scorer=dataset)
    if mode == "model":
        scorer = instantiate(cfg.oracle.scorer)
        return Oracle(scorer=scorer)
    raise ValueError(f"Unknown oracle.mode: {mode!r}. Expected 'dataset' or 'model'.")


def build_task(cfg: DictConfig) -> BaseTask:
    """Build a BaseTask from the task config group.

    Args:
        cfg: Root Hydra DictConfig containing a `task` key.

    Returns:
        Constructed task instance.

    Raises:
        ValueError: If task.type is unknown.
    """
    tcfg = cfg.task
    task_type = tcfg.type
    if task_type == "design":
        return DesignTask(
            num_acq_rounds=tcfg.num_acq_rounds,
            acq_batch_size=tcfg.acq_batch_size,
            save_round_predictions=tcfg.get("save_round_predictions", False),
        )
    if task_type == "supervised":
        return SupervisedTask()
    if task_type == "zeroshot":
        return ZeroShotTask()
    raise ValueError(
        f"Unknown task.type: {task_type!r}. Expected 'design', 'supervised', or 'zeroshot'."
    )


def build_state_loggers(cfg: DictConfig, output_dir: Path) -> list[StateLogger]:
    """Build the list of StateLoggers for a run.

    Always includes a TerminalStateLogger and a FileStateLogger writing to output_dir.

    Args:
        cfg: Root Hydra DictConfig (reserved for future per-logger config).
        output_dir: Directory where FileStateLogger writes its outputs.

    Returns:
        List containing a TerminalStateLogger and a FileStateLogger.
    """
    return [TerminalStateLogger(), FileStateLogger(output_path=output_dir)]
