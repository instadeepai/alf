from unittest.mock import MagicMock, patch

import pytest
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
    import factory
    with patch("factory.GFP") as mock_cls:
        mock_cls.return_value = MagicMock()
        factory.build_dataset(_gfp_cfg())
    mock_cls.assert_called_once()
    config_arg = mock_cls.call_args[0][0]
    assert config_arg.name == "gfp_test"
    assert config_arg.train_ratio == 0.4
    assert config_arg.seed == 0


def test_build_dataset_unknown_class_raises():
    import factory
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
    import factory
    from alf_tools.models.gp import GPModel
    model = factory.build_model(_gp_cfg())
    assert isinstance(model, GPModel)
    assert model.model_config.kernel_type == "rbf"
    assert model.train_config.num_iterations == 10
    assert model.model_config.lengthscale_prior is not None


def test_build_model_gp_matern():
    import factory
    from alf_tools.models.gp import GPModel
    model = factory.build_model(_gp_cfg(kernel_type="matern"))
    assert isinstance(model, GPModel)
    assert model.model_config.kernel_type == "matern"


def test_build_model_mlp_returns_mlp_model():
    import factory
    from alf_tools.models.mlp import MLPModel
    model = factory.build_model(_mlp_cfg())
    assert isinstance(model, MLPModel)
    assert model.model_config.hidden_dims == [64, 32]
    assert model.train_config.num_epochs == 5


def test_build_model_unknown_class_raises():
    import factory
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
    import factory
    from alf_tools.models.ensemble import EnsembleWrapper
    model = factory.build_model(_ensemble_mlp_cfg())
    assert isinstance(model, EnsembleWrapper)
    assert len(model.members) == 3


def test_build_model_ensemble_members_have_distinct_seeds():
    import factory
    model = factory.build_model(_ensemble_mlp_cfg())
    seeds = [m.model_config.model_seed for m in model.members]
    assert seeds == [0, 1, 2]


def test_build_model_ensemble_no_subsample():
    import factory
    from alf_tools.models.ensemble import EnsembleWrapper
    cfg = _ensemble_mlp_cfg()
    cfg.model.subsample = None
    model = factory.build_model(cfg)
    assert isinstance(model, EnsembleWrapper)
    assert model.config.subsample is None
