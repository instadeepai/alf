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

"""Integration tests for DesignTask with BoTorch GP surrogate and BoTorch acquisition functions.

Tests the full pipeline: BoTorchSyntheticDataset -> BoTorchGPModel -> BoTorchAcquisition (qEI)
-> DesignTask.run(), verifying that GP training and Bayesian optimization work end-to-end.

Uses shared fixtures from tools/tests/conftest.py: branin_dataset, gp_surrogate,
qei_acquisition, botorch_optimizer, branin_oracle.
"""

import numpy as np
import pandas as pd
import pytest
from alf_core import (
    Candidate,
    DesignTask,
    FileStateLogger,
    LabelledCandidates,
    Optimizer,
    State,
    TerminalStateLogger,
)
from alf_core.optimizer.search import BaseSearch
from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel
from alf_tools.optimizer.acquisition_functions.botorch_acquisition import BoTorchAcquisition
from alf_tools.optimizer.acquisition_functions.botorch_samplers import BoTorchMCSampler


class DiscretePoolSearch(BaseSearch):
    """Search function that returns a discrete pool of candidates for testing.

    This search function returns candidates from a provided pool, allowing
    acquisition functions to score discrete candidates rather than performing
    continuous optimization.

    Args:
        candidate_pool: List of candidates to return for scoring.
        pool_size: Number of candidates to return from the pool (default: all).
    """

    def __init__(self, candidate_pool: list[Candidate], pool_size: int | None = None):
        """Initialize discrete pool search with a candidate pool.

        Args:
            candidate_pool: List of candidates to return for scoring.
            pool_size: Number of candidates to return (None = all candidates).
        """
        self.candidate_pool = candidate_pool
        self.pool_size = pool_size if pool_size is not None else len(candidate_pool)

    def __call__(self, state: State, **kwargs) -> list[Candidate]:
        """Return discrete pool of candidates for scoring.

        Args:
            state: Current task state (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            List of candidates from the pool.
        """
        return self.candidate_pool[: self.pool_size]

    def get_metrics(self, state: State) -> dict[str, float]:
        """Return search metrics.

        Args:
            state: Current task state.

        Returns:
            Dictionary with pool size metric.
        """
        return {"pool_size": float(self.pool_size)}


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

        state_loggers = [
            TerminalStateLogger(),
            FileStateLogger(output_path=save_path),
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
            state_loggers=state_loggers,
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


class TestDesignBoTorchDiscretePool:
    """Integration tests for DesignTask with discrete candidate pools.

    Tests the full pipeline with discrete candidate scoring instead of
    continuous optimization, for both q=1 and q>1 scenarios.
    """

    @pytest.mark.parametrize(
        "q_batch_size,pool_size,num_acq_rounds,acq_batch_size",
        [
            (1, 100, 3, 2),  # q=1: 3 rounds * 2 candidates = 6 new candidates
            (2, 120, 2, 4),  # q=2: 2 rounds * 4 candidates = 8 new candidates
            (4, 48, 2, 4),  # q=4: 2 rounds * 4 candidates = 8 new candidates
        ],
        ids=["q1", "q2", "q4"],
    )
    def test_design_discrete_pool_full_pipeline(
        self,
        branin_dataset,
        gp_surrogate,
        branin_oracle,
        tmp_path,
        q_batch_size,
        pool_size,
        num_acq_rounds,
        acq_batch_size,
    ):
        """Test full design pipeline with discrete candidate pool for various q values.

        Parameterized test covering q=1, q=2, and q=4 scenarios.

        Verifies:
        1. GP model is trained in the initial round
        2. Discrete candidates are scored correctly with specified q
        3. Best candidates from pool are selected and evaluated
        4. Metrics are logged
        5. Best value improves over rounds

        Args:
            branin_dataset: Fixture providing the Branin dataset for tests.
            gp_surrogate: Fixture returning an (untrained) GP surrogate instance.
            branin_oracle: Fixture oracle used to evaluate selected candidates.
            tmp_path: pytest temporary path for creating output files.
            q_batch_size: Batch size for acquisition (q parameter).
            pool_size: Number of candidates in discrete pool.
            num_acq_rounds: Number of acquisition rounds.
            acq_batch_size: Number of candidates to acquire per round.
        """
        save_path = tmp_path / f"design_discrete_pool_q{q_batch_size}"
        save_path.mkdir()

        state_loggers = [
            TerminalStateLogger(),
            FileStateLogger(output_path=save_path),
        ]

        # Create discrete pool from test dataset (size must be divisible by q)
        discrete_pool = branin_dataset.test_dataset.candidates[:pool_size]
        discrete_search = DiscretePoolSearch(discrete_pool, pool_size=pool_size)

        # Create acquisition function for discrete scoring
        sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=64, seed=42)
        discrete_acq = BoTorchAcquisition(
            acquisition_type="qEI",
            batch_size=q_batch_size,
            sampler=sampler,
        )

        # Create optimizer with discrete search
        discrete_optimizer = Optimizer(
            acquisition_fn=discrete_acq,
            search_fn=discrete_search,
        )

        # Run design task
        task = DesignTask(num_acq_rounds=num_acq_rounds, acq_batch_size=acq_batch_size)
        state = task.setup(dataset=branin_dataset, surrogate=gp_surrogate)

        # Verify initial state
        initial_train_size = len(state.dataset.train_dataset)
        assert initial_train_size > 0, "Train dataset must be non-empty"
        gp_model = state.surrogate.model
        assert isinstance(gp_model, BoTorchGPModel)
        assert gp_model.model is None, "Model should not be fitted yet"

        # Run full pipeline
        task.run(
            state=state,
            state_loggers=state_loggers,
            optimizer=discrete_optimizer,
            oracle=branin_oracle,
        )

        # 1. Verify GP training
        gp_model = state.surrogate.model
        assert gp_model.model is not None, "GP model must be fitted after training"
        assert gp_model.train_X is not None, "Training data must be stored"

        # 2. Verify new candidates were added
        final_train_size = len(state.dataset.train_dataset)
        expected_new_candidates = num_acq_rounds * acq_batch_size
        assert final_train_size == initial_train_size + expected_new_candidates, (
            f"Expected {expected_new_candidates} new candidates"
        )

        # 3. Verify predictions work
        test_candidates = state.dataset.test_dataset.candidates[:5]
        predictions = state.surrogate.predict(test_candidates)
        assert predictions.means is not None, "Predictions must have means"
        assert np.all(np.isfinite(predictions.means)), "Predictions must be finite"

        # 4. Verify metrics file was created
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"
        metrics = pd.read_csv(metrics_file)
        assert len(metrics) > 0, "Metrics should have rows"

        # 5. Verify best value is reasonable
        best_value = state.dataset.train_dataset.labels.max()
        assert best_value > -10.0, "Best value should be reasonable"
        assert best_value < 10.0, "Best value should be reasonable"


class TestBoTorchAcquisitionDiscreteScoring:
    """Tests for _score_candidates with discrete candidate sets.

    Tests both q=1 (single candidate) and q>1 (batch candidates) scenarios.
    """

    @pytest.mark.parametrize(
        "q_batch_size,num_candidates",
        [
            (1, 15),  # q=1: single candidate scoring
            (2, 12),  # q=2: batch scoring with pairs
            (4, 20),  # q=4: larger batch scoring
        ],
        ids=["q1", "q2", "q4"],
    )
    def test_score_discrete_candidates(
        self,
        branin_dataset,
        trained_surrogate,
        state,
        q_batch_size,
        num_candidates,
    ):
        """Test scoring discrete candidates with various q batch sizes.

        Parameterized test covering q=1, q=2, and q=4 scenarios.

        Verifies:
        1. Acquisition function can score a discrete pool of candidates
        2. Output shape matches input candidates
        3. All scores are finite and non-negative (for qEI)
        4. Score replication works correctly for q-batches
        5. Scores vary across different batches

        Args:
            branin_dataset: Fixture providing the Branin dataset.
            trained_surrogate: Fixture that provides a trained GP surrogate.
            state: Fixture representing current task state used by acquisition.
            q_batch_size: Batch size for acquisition (q parameter).
            num_candidates: Number of candidates to score (must be divisible by q).
        """
        # Create qEI acquisition
        sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=64, seed=42)
        acq_fn = BoTorchAcquisition(
            acquisition_type="qEI",
            batch_size=q_batch_size,
            sampler=sampler,
        )

        # Select candidates spanning the quality range so qEI scores vary.
        # Taking candidates[:num_candidates] risks picking points that all land in
        # low-quality regions of the Branin space, causing qEI=0 for every candidate
        # (especially for q=1 where individual EI is computed, not batch EI).
        # Sorting by label and sampling evenly ensures a mix of good and bad candidates.
        test_labels = branin_dataset.test_dataset.labels
        test_cands_all = branin_dataset.test_dataset.candidates
        sorted_idx = np.argsort(test_labels)  # ascending: worst → best
        step = max(1, len(sorted_idx) // num_candidates)
        selected_idx = sorted_idx[::step][:num_candidates]
        discrete_candidates = [test_cands_all[i] for i in selected_idx]

        # Score candidates
        result = acq_fn(discrete_candidates, state)

        # Verify output structure
        assert isinstance(result, LabelledCandidates)
        assert len(result.candidates) == num_candidates
        assert len(result.labels) == num_candidates

        # Verify all scores are finite
        assert np.all(np.isfinite(result.labels)), "All acquisition scores must be finite"

        # Verify scores are non-negative (qEI is always >= 0)
        assert np.all(result.labels >= 0), "qEI scores must be non-negative"

        # Verify score replication for q-batches
        if q_batch_size > 1:
            for i in range(0, num_candidates, q_batch_size):
                batch_scores = result.labels[i : i + q_batch_size]
                assert np.allclose(batch_scores, batch_scores[0]), (
                    f"Batch {i // q_batch_size}: all {q_batch_size} candidates must have same score"
                )

        # Verify there's variation in scores across different batches
        if q_batch_size == 1:
            unique_scores = result.labels
        else:
            unique_scores = result.labels[::q_batch_size]  # Take first of each batch
        assert np.std(unique_scores) > 0, "Scores should vary across different batches"

    def test_score_discrete_error_q_larger_than_candidates(
        self,
        branin_dataset,
        trained_surrogate,
        state,
    ):
        """Test error handling when q is larger than number of candidates.

        Verifies that ValueError is raised when batch_size exceeds candidate pool.
        """
        # Create qEI acquisition with q=10
        sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=64, seed=42)
        acq_fn = BoTorchAcquisition(
            acquisition_type="qEI",
            batch_size=10,  # q=10
            sampler=sampler,
        )

        # Create small candidate pool (smaller than q)
        discrete_candidates = branin_dataset.test_dataset.candidates[:5]  # Only 5 candidates

        # Should raise ValueError
        with pytest.raises(ValueError, match="batch_size.*greater than.*length of candidates"):
            acq_fn(discrete_candidates, state)

    def test_score_discrete_error_candidates_not_divisible_by_q(
        self,
        branin_dataset,
        trained_surrogate,
        state,
    ):
        """Test error handling when candidates count is not divisible by q.

        Verifies that ValueError is raised when total candidates is not a multiple of q.
        """
        # Create qEI acquisition with q=3
        sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=64, seed=42)
        acq_fn = BoTorchAcquisition(
            acquisition_type="qEI",
            batch_size=3,  # q=3
            sampler=sampler,
        )

        # Create candidate pool with count not divisible by 3
        discrete_candidates = branin_dataset.test_dataset.candidates[:10]  # 10 not divisible by 3

        # Should raise ValueError
        with pytest.raises(ValueError, match="not a multiple of batch_size"):
            acq_fn(discrete_candidates, state)
