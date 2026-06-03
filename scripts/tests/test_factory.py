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

from unittest.mock import MagicMock, patch

import factory
import pytest
from alf_core.optimizer.optimizer import Optimizer
from alf_core.oracle.oracle import Oracle
from alf_core.tasks.design_task import DesignTask
from alf_core.tasks.supervised_task import SupervisedTask
from alf_core.tasks.zeroshot_task import ZeroShotTask
from alf_core.utils.state_logger import FileStateLogger, TerminalStateLogger
from alf_tools.models.ensemble import EnsembleWrapper
from alf_tools.models.gp import GPModel
from alf_tools.models.mlp import MLPModel
from alf_tools.optimizer.acquisition_functions.ucb import UCB
from omegaconf import OmegaConf


def _gfp_cfg():
    return OmegaConf.create({
        "dataset": {
            "class_name": "gfp",
            "name": "gfp_test",
            "modality": "SEQUENCE",
            "seed": 0,
            "train_ratio": 0.4,
            "validation_frac": 0.0,
            "test_ratio": 0.1,
            "split_type": "random",
            "problem_type": "regression",
            "max_candidate_pool": None,
        }
    })


def test_build_dataset_gfp_constructs_correct_config():
    """build_dataset passes the right config fields to the GFP constructor."""
    with patch("factory.GFP") as mock_cls:
        mock_cls.return_value = MagicMock()
        factory.build_dataset(_gfp_cfg())
    mock_cls.assert_called_once()
    config_arg = mock_cls.call_args[0][0]
    assert config_arg.name == "gfp_test"
    assert config_arg.train_ratio == 0.4
    assert config_arg.seed == 0


def test_build_dataset_unknown_class_raises():
    """build_dataset raises KeyError for an unregistered class_name."""
    cfg = OmegaConf.create({"dataset": {"class_name": "unknown"}})
    with pytest.raises(KeyError):
        factory.build_dataset(cfg)


def _gp_cfg(kernel_type: str = "rbf"):
    return OmegaConf.create({
        "model": {
            "class_name": "gp",
            "name": "test_gp",
            "kernel_type": kernel_type,
            "ard": True,
            "matern_nu": 2.5,
            "mean_type": "constant",
            "featurizer_type": "one_hot",
            "lengthscale_prior": {
                "_target_": "gpytorch.priors.GammaPrior",
                "concentration": 3.0,
                "rate": 6.0,
            },
            "outputscale_prior": None,
            "noise_constraint": None,
            "train": {
                "learning_rate": 0.01,
                "num_iterations": 10,
                "optimizer_type": "adam",
                "early_stopping_patience": None,
                "early_stopping_delta": 1e-4,
                "normalise_inputs": True,
                "standardise_outputs": True,
                "log_frequency": 10,
            },
        }
    })


def _mlp_cfg():
    return OmegaConf.create({
        "model": {
            "class_name": "mlp",
            "name": "test_mlp",
            "hidden_dims": [64, 32],
            "activation": "relu",
            "norm": "none",
            "dropout": 0.0,
            "n_mc_passes": 0,
            "model_seed": 0,
            "train": {
                "learning_rate": 1e-3,
                "batch_size": 32,
                "num_epochs": 5,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "log_frequency": 5,
            },
        }
    })


def test_build_model_gp_returns_gp_model():
    """build_model returns a GPModel with correct kernel and training config."""
    model = factory.build_model(_gp_cfg())
    assert isinstance(model, GPModel)
    assert model.model_config.kernel_type == "rbf"
    assert model.train_config.num_iterations == 10
    assert model.model_config.lengthscale_prior is not None


def test_build_model_gp_matern():
    """build_model creates a GPModel with matern kernel when specified."""
    model = factory.build_model(_gp_cfg(kernel_type="matern"))
    assert isinstance(model, GPModel)
    assert model.model_config.kernel_type == "matern"


def test_build_model_mlp_returns_mlp_model():
    """build_model returns an MLPModel with correct architecture and training config."""
    model = factory.build_model(_mlp_cfg())
    assert isinstance(model, MLPModel)
    assert model.model_config.hidden_dims == [64, 32]
    assert model.train_config.num_epochs == 5


def test_build_model_unknown_class_raises():
    """build_model raises KeyError for an unregistered class_name."""
    cfg = OmegaConf.create({"model": {"class_name": "unknown"}})
    with pytest.raises(KeyError):
        factory.build_model(cfg)


def _ensemble_mlp_cfg():
    return OmegaConf.create({
        "model": {
            "class_name": "ensemble",
            "name": "ensemble_test",
            "n_members": 3,
            "base_seed": 0,
            "subsample": {"fraction": 0.8, "replace": True},
            "member": {
                "class_name": "mlp",
                "name": "mlp_member",
                "hidden_dims": [32],
                "activation": "relu",
                "norm": "none",
                "dropout": 0.0,
                "n_mc_passes": 0,
                "train": {
                    "learning_rate": 1e-3,
                    "batch_size": 32,
                    "num_epochs": 2,
                    "optimizer": "adam",
                    "weight_decay": 0.0,
                    "log_frequency": 2,
                },
            },
        }
    })


def test_build_model_ensemble_returns_ensemble_wrapper():
    """build_model returns an EnsembleWrapper with the correct number of members."""
    model = factory.build_model(_ensemble_mlp_cfg())
    assert isinstance(model, EnsembleWrapper)
    assert len(model.members) == 3


def test_build_model_ensemble_members_have_distinct_seeds():
    """Each ensemble member is assigned a unique sequential seed."""
    model = factory.build_model(_ensemble_mlp_cfg())
    seeds = [m.model_config.model_seed for m in model.members]
    assert seeds == [0, 1, 2]


def test_build_model_ensemble_no_subsample():
    """build_model builds an EnsembleWrapper with no subsampling when subsample is None."""
    cfg = _ensemble_mlp_cfg()
    cfg.model.subsample = None
    model = factory.build_model(cfg)
    assert isinstance(model, EnsembleWrapper)
    assert model.config.subsample is None


def _ucb_cfg():
    return OmegaConf.create({
        "optimizer": {
            "acquisition_fn": {
                "_target_": "alf_tools.optimizer.acquisition_functions.ucb.UCB",
                "alpha": 0.9,
            },
            "search_fn": {
                "_target_": "alf_core.optimizer.search.DatasetSearch",
            },
        }
    })


def test_build_optimizer_ucb():
    """build_optimizer instantiates an Optimizer with a UCB acquisition function."""
    opt = factory.build_optimizer(_ucb_cfg())
    assert isinstance(opt, Optimizer)
    assert isinstance(opt.acquisition_fn, UCB)
    assert opt.acquisition_fn.alpha == 0.9


def test_build_oracle_dataset_mode():
    """build_oracle wraps the dataset as an Oracle scorer in dataset mode."""
    mock_dataset = MagicMock()
    cfg = OmegaConf.create({"oracle": {"mode": "dataset"}})
    oracle = factory.build_oracle(cfg, mock_dataset)
    assert isinstance(oracle, Oracle)
    assert oracle.scorer is mock_dataset


def test_build_oracle_model_mode():
    """build_oracle instantiates a scorer via _target_ in model mode."""
    cfg = OmegaConf.create({
        "oracle": {
            "mode": "model",
            "scorer": {
                "_target_": "alf_tools.optimizer.acquisition_functions.greedy.Greedy",
            },
        }
    })
    oracle = factory.build_oracle(cfg, dataset=None)
    assert isinstance(oracle, Oracle)


def test_build_task_design():
    """build_task returns a DesignTask with the configured number of rounds and batch size."""
    cfg = OmegaConf.create({
        "task": {
            "type": "design",
            "num_acq_rounds": 5,
            "acq_batch_size": 10,
            "save_round_predictions": False,
        }
    })
    task = factory.build_task(cfg)
    assert isinstance(task, DesignTask)
    assert task.num_acq_rounds == 5
    assert task.acq_batch_size == 10


def test_build_task_supervised():
    """build_task returns a SupervisedTask for type='supervised'."""
    cfg = OmegaConf.create({"task": {"type": "supervised"}})
    task = factory.build_task(cfg)
    assert isinstance(task, SupervisedTask)


def test_build_task_zeroshot():
    """build_task returns a ZeroShotTask for type='zeroshot'."""
    cfg = OmegaConf.create({"task": {"type": "zeroshot"}})
    task = factory.build_task(cfg)
    assert isinstance(task, ZeroShotTask)


def test_build_state_loggers_includes_file_and_terminal(tmp_path):
    """build_state_loggers returns both TerminalStateLogger and FileStateLogger."""
    cfg = OmegaConf.create({})
    loggers = factory.build_state_loggers(cfg, tmp_path)
    types = {type(lg) for lg in loggers}
    assert TerminalStateLogger in types
    assert FileStateLogger in types
