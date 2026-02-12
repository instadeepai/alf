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

"""Integration tests for DesignTask with BoTorch GP surrogate and BoTorch acquisition functions.

Tests the full pipeline: BoTorchSyntheticDataset -> BoTorchGPModel -> BoTorchAcquisition (qEI)
-> DesignTask.run(), verifying that GP training and Bayesian optimization work end-to-end.

Uses shared fixtures from tools/tests/conftest.py: branin_dataset, gp_surrogate,
qei_acquisition, botorch_optimizer, branin_oracle.
"""

import numpy as np
import pandas as pd
from alf_core import DesignTask, FileTaskStateLogger, TerminalTaskStateLogger
from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel


class TestDesignBoTorchSurrogate:
    """Integration tests for DesignTask with BoTorch GP and qEI acquisition."""

    def test_design_botorch_surrogate_full_pipeline(
        self,
        branin_dataset,
        gp_surrogate,
        botorch_optimizer,
        branin_oracle,
        tmp_path,
    ):
        """Test full design pipeline: GP training, acquisition, and multi-round optimization.

        Verifies:
        1. GP model is trained in the initial round (run_initial_train_round)
        2. Acquisition and oracle evaluation work correctly
        3. Metrics are logged
        4. Best value is reasonable (within function range)
        """
        save_path = tmp_path / "design_botorch_surrogate"
        save_path.mkdir()

        task_state_loggers = [
            TerminalTaskStateLogger(),
            FileTaskStateLogger(output_path=save_path),
        ]

        task = DesignTask(num_acq_rounds=3, acq_batch_size=2)
        state = task.setup(dataset=branin_dataset, surrogate=gp_surrogate)

        # Verify initial state before run
        assert len(state.dataset.train_dataset) > 0, "Train dataset must be non-empty"
        gp_model = state.surrogate.model
        assert isinstance(gp_model, BoTorchGPModel)
        assert gp_model.model is None, "Model should not be fitted yet"

        # Run design task (includes initial train round + acquisition rounds)
        task.run(
            state=state,
            task_state_loggers=task_state_loggers,
            optimizer=botorch_optimizer,
            oracle=branin_oracle,
        )

        # 1. Verify GP training: model must be fitted after run_initial_train_round
        gp_model = state.surrogate.model
        assert isinstance(gp_model, BoTorchGPModel)
        assert gp_model.model is not None, "GP model must be fitted after training"
        assert gp_model.train_X is not None, "Training data must be stored"

        # 2. Verify training metrics were recorded (loss should have been optimized)
        training_metrics = gp_model._training_metrics
        assert "loss" in training_metrics, "Training loss should be recorded"
        assert len(training_metrics["loss"]) > 0, "At least one loss value should exist"

        # 3. Verify surrogate can make predictions
        test_candidates = state.dataset.test_dataset.candidates[:3]
        predictions = state.surrogate.predict(test_candidates)
        assert predictions.means is not None, "Predictions must have means"
        assert len(predictions.means) == len(test_candidates), "Prediction count must match"

        # 4. Verify metrics file was created
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"
        metrics = pd.read_csv(metrics_file)
        assert len(metrics) > 0, "Metrics should have rows"

        # 5. Verify best value is within reasonable range (Branin optimum ~0.398 after negate)
        best_value = state.dataset.train_dataset.labels.max()
        assert best_value > -10.0, "Best value should be reasonable (Branin negated max ~0.4)"
        assert best_value < 10.0, "Best value should be reasonable"

    def test_initial_train_round_fits_gp(
        self,
        branin_dataset,
        gp_surrogate,
        tmp_path,
    ):
        """Test that run_initial_train_round correctly trains the GP on train/val data."""
        task = DesignTask(num_acq_rounds=0, acq_batch_size=2)
        state = task.setup(dataset=branin_dataset, surrogate=gp_surrogate)

        gp_model = state.surrogate.model
        assert isinstance(gp_model, BoTorchGPModel)
        assert gp_model.model is None

        # Run only the initial train round (task.run would do this, but we test it directly)
        state = task.run_initial_train_round(
            state=state,
            state_loggers=[],
        )

        # GP must be fitted
        gp_model = state.surrogate.model
        assert isinstance(gp_model, BoTorchGPModel)
        assert gp_model.model is not None
        assert gp_model.train_X is not None
        assert len(gp_model._training_metrics["loss"]) > 0

        # Predictions on train data should be finite
        train_preds = state.surrogate.predict(state.dataset.train_dataset.candidates[:5])
        assert np.all(np.isfinite(train_preds.means)), "Train predictions must be finite"
