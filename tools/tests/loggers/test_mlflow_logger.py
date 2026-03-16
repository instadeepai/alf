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

from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_tools.loggers.mlflow_logger import MLflowLogger


def make_state(round_val, metrics, training_history=None):
    """Build a minimal state-like mock for MLflowLogger tests.

    Returns:
        A MagicMock with round_metrics set to a RoundMetrics instance.
    """
    state = MagicMock()
    state.round_metrics = RoundMetrics(
        round=round_val,
        metrics=metrics,
        training_history=training_history or [],
    )
    return state


class TestMLflowLogger:
    """Tests for MLflowLogger."""

    def test_log_calls_mlflow_log_metrics_for_round_metrics(self):
        """log() must call mlflow.log_metrics at least once."""
        with patch("alf_tools.loggers.mlflow_logger.mlflow") as mock_mlflow:
            ml_logger = MLflowLogger()
            state = make_state(round_val=2, metrics={"tell_time": 1.0, "ask_time": 0.5})
            ml_logger.log(state)
            assert mock_mlflow.log_metrics.called

    def test_log_round_metrics_uses_round_prefix_and_correct_step(self):
        """Round metrics must be logged with step=round_metrics.round."""
        with patch("alf_tools.loggers.mlflow_logger.mlflow") as mock_mlflow:
            ml_logger = MLflowLogger()
            state = make_state(round_val=4, metrics={"tell_time": 0.8})
            ml_logger.log(state)
            calls = mock_mlflow.log_metrics.call_args_list
            round_calls = [c for c in calls if c[1].get("step") == 4]
            assert len(round_calls) >= 1

    def test_log_training_history_uses_epoch_step(self):
        """Epoch metrics must be logged with step values matching each epoch index."""
        with patch("alf_tools.loggers.mlflow_logger.mlflow") as mock_mlflow:
            ml_logger = MLflowLogger()
            em0 = EpochMetrics(epoch=0, train_loss=0.9)
            em1 = EpochMetrics(epoch=1, train_loss=0.7)
            state = make_state(round_val=1, metrics={}, training_history=[em0, em1])
            ml_logger.log(state)
            calls = mock_mlflow.log_metrics.call_args_list
            steps_used = {c[1].get("step") for c in calls if "step" in c[1]}
            assert 0 in steps_used
            assert 1 in steps_used

    def test_empty_training_history_does_not_raise(self):
        """log() with an empty training_history must not raise."""
        with patch("alf_tools.loggers.mlflow_logger.mlflow"):
            ml_logger = MLflowLogger()
            state = make_state(round_val=1, metrics={"tell_time": 0.1})
            ml_logger.log(state)

    def test_log_accepts_optional_round_name(self):
        """log() must accept an optional round_name without raising."""
        with patch("alf_tools.loggers.mlflow_logger.mlflow"):
            ml_logger = MLflowLogger()
            state = make_state(round_val=0, metrics={})
            ml_logger.log(state, round_name="initial_train_round")
