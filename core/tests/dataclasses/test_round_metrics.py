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

from alf_core.dataclasses import EpochMetrics, RoundMetrics
from alf_core.dataclasses.epoch_metrics import EpochMetrics as EpochMetricsModule
from alf_core.dataclasses.round_metrics import RoundMetrics as RoundMetricsModule


class TestRoundMetrics:
    """Test suite for RoundMetrics dataclass."""

    def test_construction_with_round_only(self) -> None:
        """Test RoundMetrics construction with only round parameter."""
        rm = RoundMetrics(round=3)
        assert rm.round == 3
        assert rm.metrics == {}
        assert rm.training_history == []

    def test_construction_with_all_fields(self) -> None:
        """Test RoundMetrics construction with all fields provided."""
        em = EpochMetrics(epoch=0, train_loss=0.5)
        rm = RoundMetrics(round=1, metrics={"tell_time": 1.2}, training_history=[em])
        assert rm.round == 1
        assert rm.metrics["tell_time"] == 1.2
        assert len(rm.training_history) == 1

    def test_metrics_dict_is_mutable(self) -> None:
        """Test that metrics dict can be modified after construction."""
        rm = RoundMetrics(round=0)
        rm.metrics["ask_time"] = 0.5
        assert rm.metrics["ask_time"] == 0.5

    def test_training_history_is_mutable(self) -> None:
        """Test that training_history list can be modified after construction."""
        rm = RoundMetrics(round=0)
        em = EpochMetrics(epoch=0, train_loss=0.9)
        rm.training_history = [em]
        assert len(rm.training_history) == 1

    def test_default_metrics_not_shared_between_instances(self) -> None:
        """Test that default metrics dict is not shared between instances."""
        rm1 = RoundMetrics(round=0)
        rm2 = RoundMetrics(round=1)
        rm1.metrics["key"] = "value"
        assert "key" not in rm2.metrics

    def test_exported_from_alf_core_dataclasses(self) -> None:
        """Test that RoundMetrics and EpochMetrics are exported from alf_core.dataclasses."""
        # Import at the top level verifies exports are available
        assert EpochMetrics is EpochMetricsModule
        assert RoundMetrics is RoundMetricsModule
