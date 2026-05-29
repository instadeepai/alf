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

from dataclasses import dataclass, fields, is_dataclass

import torch
from alf_core.model.base_model import BaseTrainConfig


class TestBaseTrainConfig:
    """Tests for BaseTrainConfig."""

    def test_default_values(self):
        """Default learning_rate and log_frequency must match expected values."""
        config = BaseTrainConfig()
        assert config.learning_rate == 1e-3
        assert config.log_frequency == 10

    def test_custom_values(self):
        """Custom values must be stored on the config instance."""
        config = BaseTrainConfig(learning_rate=0.01, log_frequency=5)
        assert config.learning_rate == 0.01
        assert config.log_frequency == 5

    def test_is_dataclass(self):
        """BaseTrainConfig must be a dataclass with learning_rate and log_frequency fields."""
        assert is_dataclass(BaseTrainConfig)
        config = BaseTrainConfig()
        field_names = {f.name for f in fields(config)}
        assert "learning_rate" in field_names
        assert "log_frequency" in field_names

    def test_label_dtype_defaults_to_none(self):
        """label_dtype must default to None (model resolves its own default)."""
        config = BaseTrainConfig()
        assert config.label_dtype is None

    def test_label_dtype_can_be_set(self):
        """label_dtype must accept a torch.dtype value."""
        config = BaseTrainConfig(label_dtype=torch.float64)
        assert config.label_dtype == torch.float64

    def test_subclass_inherits_fields(self) -> None:
        """Subclass must inherit learning_rate and log_frequency from BaseTrainConfig."""

        @dataclass
        class ConcreteTrainConfig(BaseTrainConfig):
            extra_field: int = 99

        config = ConcreteTrainConfig()
        assert config.learning_rate == 1e-3
        assert config.log_frequency == 10
        assert config.extra_field == 99

    def test_subclass_inherits_label_dtype(self) -> None:
        """Subclass must inherit label_dtype from BaseTrainConfig."""

        @dataclass
        class ConcreteTrainConfig(BaseTrainConfig):
            extra_field: int = 0

        assert ConcreteTrainConfig().label_dtype is None
        assert ConcreteTrainConfig(label_dtype=torch.long).label_dtype == torch.long
