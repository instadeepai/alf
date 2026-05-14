from dataclasses import fields
import pytest
from alf_core.model.base_train_config import BaseTrainConfig


class TestBaseTrainConfig:
    """Tests for BaseTrainConfig."""

    def test_default_values(self):
        config = BaseTrainConfig()
        assert config.learning_rate == 1e-3
        assert config.log_frequency == 10

    def test_custom_values(self):
        config = BaseTrainConfig(learning_rate=0.01, log_frequency=5)
        assert config.learning_rate == 0.01
        assert config.log_frequency == 5

    def test_is_dataclass(self):
        config = BaseTrainConfig()
        field_names = {f.name for f in fields(config)}
        assert "learning_rate" in field_names
        assert "log_frequency" in field_names

    def test_subclass_inherits_fields(self):
        from dataclasses import dataclass

        @dataclass
        class ConcreteTrainConfig(BaseTrainConfig):
            extra_field: int = 99

        config = ConcreteTrainConfig()
        assert config.learning_rate == 1e-3
        assert config.log_frequency == 10
        assert config.extra_field == 99
