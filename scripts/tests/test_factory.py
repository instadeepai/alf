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
