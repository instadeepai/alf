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

"""Tests for TerminalStateLogger and FileStateLogger."""

import logging
import types
from typing import Any

import pandas as pd
from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.utils.state_logger import FileStateLogger, TerminalStateLogger


def make_state_stub(round_val: int, metrics: dict, training_history: Any = None) -> Any:
    """Build a minimal state-like object for logger tests.

    Returns:
        A SimpleNamespace instance with round_metrics, round_predictions,
        history, dataset, and round attributes populated.
    """
    dataset_stub = types.SimpleNamespace(
        test_dataset=types.SimpleNamespace(candidates=[], labels=[])
    )
    return types.SimpleNamespace(
        round_metrics=RoundMetrics(
            round=round_val,
            metrics=metrics,
            training_history=training_history or [],
        ),
        round_predictions=None,
        history=[],
        dataset=dataset_stub,
        round=round_val,
    )


class TestTerminalStateLogger:
    """Tests for TerminalStateLogger."""

    def test_log_uses_provided_round_name(self, caplog: Any) -> None:
        """Verify that the provided round_name appears in the log output."""
        logger_inst = TerminalStateLogger()
        state = make_state_stub(round_val=2, metrics={"tell_time": 1.5})
        with caplog.at_level(logging.INFO, logger="alf-core"):
            logger_inst.log(state, round_name="test_round")
        assert "test_round" in caplog.text

    def test_log_fallback_round_name_uses_round_metrics_round(self, caplog: Any) -> None:
        """When round_name is None, fallback must use round_metrics.round."""
        logger_inst = TerminalStateLogger()
        state = make_state_stub(round_val=5, metrics={"val": 1.0})
        with caplog.at_level(logging.INFO, logger="alf-core"):
            logger_inst.log(state)  # round_name=None → fallback
        assert "5" in caplog.text

    def test_log_fallback_does_not_use_state_round_when_diverged(self, caplog: Any) -> None:
        """After state.update(), state.round is round+1. Fallback must use round_metrics.round."""
        logger_inst = TerminalStateLogger()
        state = make_state_stub(round_val=3, metrics={"val": 1.0})
        state.round = 4  # simulates state.round having been incremented by state.update()
        with caplog.at_level(logging.INFO, logger="alf-core"):
            logger_inst.log(state)
        assert "Round 3" in caplog.text
        assert "Round 4" not in caplog.text

    def test_log_iterates_metrics_not_round_metrics_object(self, caplog: Any) -> None:
        """Each metric key must appear in the log output."""
        logger_inst = TerminalStateLogger()
        state = make_state_stub(round_val=1, metrics={"tell_time": 2.5, "ask_time": 0.3})
        with caplog.at_level(logging.INFO, logger="alf-core"):
            logger_inst.log(state, round_name="round_1")
        assert "tell_time" in caplog.text
        assert "ask_time" in caplog.text


class TestFileStateLogger:
    """Tests for FileStateLogger."""

    def test_log_writes_round_metrics_dot_metrics_to_csv(self, tmp_path: Any) -> None:
        """round_metrics.metrics keys must be written as columns in metrics.csv."""
        fl = FileStateLogger(output_path=tmp_path)
        state = make_state_stub(round_val=1, metrics={"tell_time": 2.0, "ask_time": 0.5})
        fl.log(state, round_name="round_1")
        df = pd.read_csv(tmp_path / "metrics.csv")
        assert "tell_time" in df.columns
        assert "ask_time" in df.columns

    def test_log_does_not_write_round_key(self, tmp_path: Any) -> None:
        """Round is in RoundMetrics.round, not in .metrics — it must not appear in CSV."""
        fl = FileStateLogger(output_path=tmp_path)
        state = make_state_stub(round_val=2, metrics={"tell_time": 1.0})
        fl.log(state, round_name="round_2")
        df = pd.read_csv(tmp_path / "metrics.csv")
        assert "round" not in df.columns

    def test_log_fallback_round_name_uses_round_metrics_round(self, tmp_path: Any) -> None:
        """When round_name is None, fallback must use round_metrics.round without raising."""
        fl = FileStateLogger(output_path=tmp_path)
        state = make_state_stub(round_val=7, metrics={"val": 1.0})
        fl.log(state)  # no round_name → fallback; must not raise

    def test_training_history_not_written_to_csv(self, tmp_path: Any) -> None:
        """Epoch-level columns from training_history must not appear in metrics.csv."""
        fl = FileStateLogger(output_path=tmp_path)
        em = EpochMetrics(epoch=0, train_loss=0.5)
        state = make_state_stub(round_val=1, metrics={"tell_time": 1.0}, training_history=[em])
        fl.log(state, round_name="round_1")
        df = pd.read_csv(tmp_path / "metrics.csv")
        # No epoch-level columns should appear in metrics.csv
        assert "epoch" not in df.columns
        assert "train_loss" not in df.columns
