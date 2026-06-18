# Copyright 2026 InstaDeep Ltd. All rights reserved.
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

from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics


class TestSurrogateEpochMetrics:
    """Test suite for SurrogateEpochMetrics dataclass."""

    def test_to_metrics_dict_required_fields_always_present(self) -> None:
        """Test that required fields are always present in metrics dict."""
        em = SurrogateEpochMetrics(epoch=3, train_loss=0.5)
        d = em.to_metrics_dict()
        assert d["epoch"] == 3.0
        assert d["train_loss"] == 0.5

    def test_to_metrics_dict_none_fields_excluded(self) -> None:
        """Test that None fields are excluded from metrics dict."""
        em = SurrogateEpochMetrics(epoch=1, train_loss=0.8)
        d = em.to_metrics_dict()
        assert "val_loss" not in d

    def test_to_metrics_dict_optional_fields_included_when_set(self) -> None:
        """Test that optional fields are included when set."""
        em = SurrogateEpochMetrics(
            epoch=2,
            train_loss=0.4,
            val_loss=0.6,
            additional_metrics={
                "train_spearman": 0.9,
                "val_spearman": 0.85,
                "train_mse": 0.1,
                "val_mse": 0.15,
            },
        )
        d = em.to_metrics_dict()
        assert d["val_loss"] == 0.6
        assert d["train_spearman"] == 0.9
        assert d["val_spearman"] == 0.85
        assert d["train_mse"] == 0.1
        assert d["val_mse"] == 0.15

    def test_to_metrics_dict_additional_metrics_merged(self) -> None:
        """Test that additional_metrics are merged into the dict."""
        em = SurrogateEpochMetrics(
            epoch=1, train_loss=0.5, additional_metrics={"custom_metric": 42.0}
        )
        d = em.to_metrics_dict()
        assert d["custom_metric"] == 42.0

    def test_epoch_field_converted_to_float(self) -> None:
        """Test that epoch is converted to float in metrics dict."""
        em = SurrogateEpochMetrics(epoch=5, train_loss=0.3)
        d = em.to_metrics_dict()
        assert isinstance(d["epoch"], int)
