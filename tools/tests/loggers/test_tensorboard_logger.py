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

from unittest.mock import MagicMock

from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_tools.loggers.tensorboard_logger import TensorBoardLogger


def make_state(round_val, metrics, training_history=None):
    """Build a minimal state-like mock for TensorBoardLogger tests.

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


class TestTensorBoardLogger:
    """Tests for TensorBoardLogger."""

    def test_log_calls_add_scalar(self):
        """log() must call writer.add_scalar at least once."""
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=2, metrics={"tell_time": 1.0})
        tb_logger.log(state)
        assert mock_writer.add_scalar.called

    def test_log_round_metrics_uses_round_prefix_and_correct_step(self):
        """Round metrics must be written under 'round/' with step=round_metrics.round."""
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=3, metrics={"tell_time": 0.5, "ask_time": 0.2})
        tb_logger.log(state)
        calls = mock_writer.add_scalar.call_args_list
        tags = [c[0][0] for c in calls]
        assert "round/tell_time" in tags
        assert "round/ask_time" in tags
        round_calls = [c for c in calls if c[0][0].startswith("round/")]
        for c in round_calls:
            assert c[0][2] == 3

    def test_log_training_history_uses_training_prefix_and_epoch_step(self):
        """Epoch metrics must be written under 'training/' with step=epoch."""
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        em0 = EpochMetrics(epoch=0, train_loss=0.9)
        em1 = EpochMetrics(epoch=1, train_loss=0.7)
        state = make_state(round_val=1, metrics={}, training_history=[em0, em1])
        tb_logger.log(state)
        calls = mock_writer.add_scalar.call_args_list
        training_calls = [c for c in calls if c[0][0].startswith("training/")]
        assert len(training_calls) >= 2
        for c in training_calls:
            tag, value, step = c[0]
            if tag == "training/train_loss" and step == 0:
                assert abs(value - 0.9) < 1e-6

    def test_empty_training_history_does_not_raise(self):
        """log() with an empty training_history must not raise."""
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=1, metrics={"tell_time": 0.1})
        tb_logger.log(state)

    def test_log_accepts_optional_round_name(self):
        """log() must accept an optional round_name without raising."""
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=0, metrics={"val": 1.0})
        tb_logger.log(state, round_name="initial_train_round")
