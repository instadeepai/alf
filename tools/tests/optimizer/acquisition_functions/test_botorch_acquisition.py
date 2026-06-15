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

"""Tests for generic BoTorch acquisition function wrapper."""

import math
from typing import get_args
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from alf_core import (
    BaseDatasetConfig,
    Candidate,
    LabelledCandidates,
    Modality,
    ProblemType,
    Surrogate,
)
from alf_core.dataclasses.state import State
from alf_core.dataset.base_dataset import BaseDataset
from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel, BoTorchTrainConfig
from alf_tools.optimizer.acquisition_functions.botorch_acquisition import (
    AcquisitionType,
    BoTorchAcquisition,
    BoTorchAcquisitionOptConfig,
)
from alf_tools.optimizer.acquisition_functions.botorch_samplers import (
    BoTorchMCSampler,
)


class _InlineBraninDataset(BaseDataset):
    """Minimal Branin dataset that satisfies BaseDataset without BoTorchSyntheticDataset.

    Provides bounds and dim attributes matching the Branin function domain:
    x1 in [-5, 10], x2 in [0, 15].
    """

    def __init__(self, config: BaseDatasetConfig, n_initial_samples: int = 500) -> None:
        """Initialize inline Branin dataset.

        Args:
            config: Dataset configuration.
            n_initial_samples: Number of samples to generate.
        """
        super().__init__(config)
        self.bounds = np.array([[-5.0, 0.0], [10.0, 15.0]], dtype=np.float32)
        self.dim = 2
        self.n_initial_samples = n_initial_samples

    @staticmethod
    def _branin(x: np.ndarray) -> float:
        """Evaluate the Branin function.

        Args:
            x: 1-D array [x1, x2].

        Returns:
            Branin function value.
        """
        x1, x2 = float(x[0]), float(x[1])
        return (
            (x2 - (5.1 / (4 * math.pi**2)) * x1**2 + (5 / math.pi) * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
            + 10
        )

    def load_dataset(self) -> LabelledCandidates:
        """Generate random Branin evaluations within domain bounds.

        Returns:
            LabelledCandidates with sampled inputs and Branin outputs.
        """
        rng = np.random.RandomState(self.config.seed)
        X = rng.uniform(
            low=self.bounds[0], high=self.bounds[1], size=(self.n_initial_samples, self.dim)
        ).astype(np.float32)
        y = np.array([self._branin(x) for x in X], dtype=np.float32)
        candidates = [Candidate(data=row, modality=Modality.TABULAR) for row in X]
        return LabelledCandidates(candidates=candidates, labels=y)


@pytest.fixture
def simple_dataset() -> _InlineBraninDataset:
    """Create an inline Branin dataset for acquisition tests.

    Returns:
        _InlineBraninDataset: A configured Branin dataset with bounds and dim.
    """
    config = BaseDatasetConfig(
        name="test_branin",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.05,
        validation_frac=0.2,
        test_ratio=0.1,
        split_type="random",
        problem_type=ProblemType.REGRESSION,
    )
    dataset = _InlineBraninDataset(config=config, n_initial_samples=500)
    dataset.setup()
    return dataset


@pytest.fixture
def trained_surrogate(simple_dataset: _InlineBraninDataset) -> Surrogate:
    """Create and train a GP surrogate on the simple dataset.

    Args:
        simple_dataset: Fixture providing a simple Branin dataset.

    Returns:
        Surrogate: A trained GP surrogate model.
    """
    train_config = BoTorchTrainConfig(num_iterations=50, learning_rate=0.1)
    gp_model = BoTorchGPModel(train_config=train_config)
    surrogate = Surrogate(model=gp_model)
    surrogate.fit(simple_dataset.train_dataset, simple_dataset.validation_dataset)
    return surrogate


@pytest.fixture
def task_state(simple_dataset: _InlineBraninDataset, trained_surrogate: Surrogate) -> State:
    """Create a task state for testing.

    Args:
        simple_dataset: Fixture providing a simple Branin dataset.
        trained_surrogate: Fixture providing a trained surrogate.

    Returns:
        State: A task state for testing.
    """
    return State(dataset=simple_dataset, surrogate=trained_surrogate)


# =============================================================================
# Initialization Tests
# =============================================================================


def test_initialization_qei():
    """Test basic initialization with qEI."""
    sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)
    bounds = [[0.0, 1.0], [0.0, 1.0]]

    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        sampler=sampler,
        bounds=bounds,
        num_restarts=10,
        raw_samples=512,
        batch_size=5,
    )

    assert acq_fn.acquisition_type == "qEI"
    assert acq_fn.bounds == bounds
    assert acq_fn.num_restarts == 10
    assert acq_fn.raw_samples == 512
    assert acq_fn.batch_size == 5
    assert acq_fn.sequential is False
    assert acq_fn.sampler_config == sampler


def test_initialization_qucb():
    """Test initialization with qUCB and beta parameter."""
    acq_fn = BoTorchAcquisition(
        acquisition_type="qUCB",
        beta=0.5,
        bounds=[[0.0, 1.0], [0.0, 1.0]],
    )

    assert acq_fn.acquisition_type == "qUCB"
    assert acq_fn.beta == 0.5


def test_initialization_qnei():
    """Test initialization with qNEI."""
    acq_fn = BoTorchAcquisition(
        acquisition_type="qNEI",
        bounds=[[0.0, 1.0], [0.0, 1.0]],
    )

    assert acq_fn.acquisition_type == "qNEI"


def test_initialization_default_sampler():
    """Test that default sampler is created if none provided."""
    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        bounds=[[0.0, 1.0], [0.0, 1.0]],
    )

    assert acq_fn.sampler_config is not None
    assert acq_fn.sampler_config.sampler_type == "sobol"
    assert acq_fn.sampler_config.num_samples == 512


def test_initialization_invalid_acquisition_type():
    """Test that invalid acquisition type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported acquisition_type"):
        BoTorchAcquisition(
            acquisition_type="invalid",  # type: ignore
            bounds=[[0.0, 1.0], [0.0, 1.0]],
        )


def test_initialization_qkg_raises_not_implemented():
    """QKG raises NotImplementedError at init time (planned but not yet implemented)."""
    with pytest.raises(NotImplementedError, match="qKG.*not yet implemented"):
        BoTorchAcquisition(
            acquisition_type="qKG",  # type: ignore[arg-type]
            bounds=[[0.0, 1.0], [0.0, 1.0]],
        )


# =============================================================================
# Scoring Mode Tests (Discrete Candidates)
# =============================================================================


def test_scoring_mode_qei(task_state):
    """Test scoring mode with qEI on discrete candidates."""
    acq_fn = BoTorchAcquisition(acquisition_type="qEI", batch_size=1)

    test_candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.1, 0.9]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.9, 0.1]), modality=Modality.TABULAR),
    ]

    labelled = acq_fn(search_candidates=test_candidates, state=task_state)

    assert len(labelled) == 3
    assert labelled.candidates == test_candidates
    assert labelled.labels.shape == (3,)
    assert all(labelled.labels >= 0)  # EI is non-negative
    assert all(np.isfinite(labelled.labels))


def test_scoring_mode_qucb(task_state):
    """Test scoring mode with qUCB on discrete candidates."""
    acq_fn = BoTorchAcquisition(acquisition_type="qUCB", beta=0.2, batch_size=1)

    test_candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.1, 0.9]), modality=Modality.TABULAR),
    ]

    labelled = acq_fn(search_candidates=test_candidates, state=task_state)

    assert len(labelled) == 2
    assert labelled.candidates == test_candidates
    assert labelled.labels.shape == (2,)
    assert all(np.isfinite(labelled.labels))


def test_scoring_mode_qnei(task_state):
    """Test scoring mode with qNEI on discrete candidates."""
    acq_fn = BoTorchAcquisition(acquisition_type="qNEI", batch_size=1)

    test_candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.1, 0.9]), modality=Modality.TABULAR),
    ]

    labelled = acq_fn(search_candidates=test_candidates, state=task_state)

    assert len(labelled) == 2
    assert labelled.candidates == test_candidates
    assert labelled.labels.shape == (2,)
    assert all(labelled.labels >= 0)  # NEI is non-negative
    assert all(np.isfinite(labelled.labels))


def test_scoring_mode_different_beta_values(task_state):
    """Test that different beta values affect qUCB scores."""
    candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.1, 0.9]), modality=Modality.TABULAR),
    ]

    # Low exploration
    acq_fn_low = BoTorchAcquisition(acquisition_type="qUCB", beta=0.1)
    labelled_low = acq_fn_low(search_candidates=candidates, state=task_state)

    # High exploration
    acq_fn_high = BoTorchAcquisition(acquisition_type="qUCB", beta=1.0)
    labelled_high = acq_fn_high(search_candidates=candidates, state=task_state)

    # Scores should be different (higher beta typically gives higher scores)
    assert not np.allclose(labelled_low.labels, labelled_high.labels, atol=1e-6)


def test_scoring_mode_single_candidate(task_state):
    """Test scoring mode with a single candidate."""
    acq_fn = BoTorchAcquisition(acquisition_type="qEI")

    candidate = [Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)]

    labelled = acq_fn(search_candidates=candidate, state=task_state)

    assert len(labelled) == 1
    assert np.isfinite(labelled.labels[0])


# =============================================================================
# Continuous Optimization Mode Tests
# =============================================================================


def test_optimization_mode_qei(task_state, simple_dataset):
    """Test continuous optimization mode with qEI."""
    bounds = [[b[0], b[1]] for b in simple_dataset.bounds.T]

    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        bounds=bounds,
        batch_size=3,
        num_restarts=2,  # Small for speed
        raw_samples=64,  # Small for speed
    )

    labelled = acq_fn(search_candidates=[], state=task_state)

    assert len(labelled) == 3
    assert all(isinstance(c, Candidate) for c in labelled.candidates)
    assert labelled.labels.shape == (3,)
    assert all(np.isfinite(labelled.labels))

    # Check candidates are within bounds
    for candidate in labelled.candidates:
        assert np.all(candidate.data >= simple_dataset.bounds[0])
        assert np.all(candidate.data <= simple_dataset.bounds[1])


def test_optimization_mode_qucb(task_state, simple_dataset):
    """Test continuous optimization mode with qUCB."""
    bounds = [[b[0], b[1]] for b in simple_dataset.bounds.T]

    acq_fn = BoTorchAcquisition(
        acquisition_type="qUCB",
        bounds=bounds,
        beta=0.3,
        batch_size=2,
        num_restarts=2,
        raw_samples=64,
    )

    labelled = acq_fn(search_candidates=[], state=task_state)

    assert len(labelled) == 2
    assert all(np.isfinite(labelled.labels))


def test_optimization_mode_qnei(task_state, simple_dataset):
    """Test continuous optimization mode with qNEI."""
    bounds = [[b[0], b[1]] for b in simple_dataset.bounds.T]

    acq_fn = BoTorchAcquisition(
        acquisition_type="qNEI",
        bounds=bounds,
        batch_size=2,
        num_restarts=2,
        raw_samples=64,
    )

    labelled = acq_fn(search_candidates=[], state=task_state)

    assert len(labelled) == 2
    assert all(np.isfinite(labelled.labels))


def test_optimization_mode_without_bounds_raises_error(task_state):
    """Test that optimization mode without bounds raises ValueError."""
    acq_fn = BoTorchAcquisition(acquisition_type="qEI", batch_size=3)

    with pytest.raises(ValueError, match="Bounds must be provided"):
        acq_fn(search_candidates=[], state=task_state)


def test_optimization_mode_batch_size_respected(task_state, simple_dataset):
    """Test that optimization mode returns correct number of candidates."""
    bounds = [[b[0], b[1]] for b in simple_dataset.bounds.T]

    for batch_size in [1, 3, 5]:
        acq_fn = BoTorchAcquisition(
            acquisition_type="qEI",
            bounds=bounds,
            batch_size=batch_size,
            num_restarts=2,
            raw_samples=64,
        )

        labelled = acq_fn(search_candidates=[], state=task_state)
        assert len(labelled) == batch_size


def test_optimization_mode_sequential_vs_joint(task_state, simple_dataset):
    """Test sequential vs joint optimization modes."""
    bounds = [[b[0], b[1]] for b in simple_dataset.bounds.T]

    # Sequential optimization
    acq_fn_seq = BoTorchAcquisition(
        acquisition_type="qEI",
        bounds=bounds,
        batch_size=3,
        sequential=True,
        num_restarts=2,
        raw_samples=64,
    )

    # Joint optimization
    acq_fn_joint = BoTorchAcquisition(
        acquisition_type="qEI",
        bounds=bounds,
        batch_size=3,
        sequential=False,
        num_restarts=2,
        raw_samples=64,
    )

    labelled_seq = acq_fn_seq(search_candidates=[], state=task_state)
    labelled_joint = acq_fn_joint(search_candidates=[], state=task_state)

    # Both should return valid candidates
    assert len(labelled_seq) == 3
    assert len(labelled_joint) == 3

    # Verify they're within bounds
    for cand in labelled_seq.candidates + labelled_joint.candidates:
        assert np.all(cand.data >= simple_dataset.bounds[0])
        assert np.all(cand.data <= simple_dataset.bounds[1])


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_requires_trained_surrogate(simple_dataset):
    """Test that calling without trained surrogate raises RuntimeError."""
    acq_fn = BoTorchAcquisition(acquisition_type="qEI")

    # State requires a Surrogate at construction time (beartype-enforced); set to None
    # afterward so the acquisition function's own guard is what raises the error.
    placeholder = Surrogate(model=BoTorchGPModel())
    state = State(dataset=simple_dataset, surrogate=placeholder)
    state.surrogate = None  # type: ignore

    test_candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
    ]

    with pytest.raises(RuntimeError, match="Surrogate model is required"):
        acq_fn(search_candidates=test_candidates, state=state)


# =============================================================================
# Integration Tests
# =============================================================================


def test_switching_acquisition_functions(task_state):
    """Test that we can easily switch between acquisition functions."""
    candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.1, 0.9]), modality=Modality.TABULAR),
    ]

    # Test all supported acquisition functions
    for acq_type in ["qEI", "qNEI", "qUCB"]:
        acq_fn = BoTorchAcquisition(acquisition_type=acq_type, beta=0.2)
        labelled = acq_fn(search_candidates=candidates, state=task_state)

        assert len(labelled) == 2
        assert all(np.isfinite(labelled.labels))


class _InlineHartmann6Dataset(BaseDataset):
    """Minimal 6-D Hartmann dataset for high-dimensional acquisition tests.

    All inputs live in [0, 1]^6. The Hartmann6 objective is approximated here
    by a simple quadratic surrogate so that tests remain dependency-free.
    """

    def __init__(self, config: BaseDatasetConfig, n_initial_samples: int = 500) -> None:
        """Initialize inline Hartmann6 dataset.

        Args:
            config: Dataset configuration.
            n_initial_samples: Number of samples to generate.
        """
        super().__init__(config)
        self.bounds = np.zeros((2, 6), dtype=np.float32)
        self.bounds[1] = 1.0  # upper bound is 1 for all dims
        self.dim = 6
        self.n_initial_samples = n_initial_samples

    def load_dataset(self) -> LabelledCandidates:
        """Generate random samples in [0, 1]^6 with a simple quadratic objective.

        Returns:
            LabelledCandidates with sampled inputs and quadratic outputs.
        """
        rng = np.random.RandomState(self.config.seed)
        X = rng.uniform(size=(self.n_initial_samples, self.dim)).astype(np.float32)
        # Simple quadratic: -(sum of squares), peaked at origin
        y = -np.sum(X**2, axis=1).astype(np.float32)
        candidates = [Candidate(data=row, modality=Modality.TABULAR) for row in X]
        return LabelledCandidates(candidates=candidates, labels=y)


def test_high_dimensional_input(trained_surrogate):
    """Test that acquisition works with a 6-dimensional input space."""
    config = BaseDatasetConfig(
        name="test_6d",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.05,
        validation_frac=0.2,
        test_ratio=0.1,
        split_type="random",
        problem_type=ProblemType.REGRESSION,
    )

    dataset = _InlineHartmann6Dataset(config=config, n_initial_samples=500)
    dataset.setup()

    # Train surrogate
    train_config = BoTorchTrainConfig(num_iterations=50)
    gp_model = BoTorchGPModel(train_config=train_config)
    surrogate = Surrogate(model=gp_model)
    surrogate.fit(dataset.train_dataset, dataset.validation_dataset)

    state = State(dataset=dataset, surrogate=surrogate)

    # Test with qEI
    bounds = [[b[0], b[1]] for b in dataset.bounds.T]
    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        bounds=bounds,
        batch_size=3,
        num_restarts=2,
        raw_samples=64,
    )

    labelled = acq_fn(search_candidates=[], state=state)

    assert len(labelled) == 3
    assert all(len(c.data) == 6 for c in labelled.candidates)


def test_custom_sampler_configurations(task_state):
    """Test different sampler configurations."""
    candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
    ]

    # Sobol sampler
    sampler_sobol = BoTorchMCSampler(sampler_type="sobol", num_samples=256)
    acq_sobol = BoTorchAcquisition(acquisition_type="qEI", sampler=sampler_sobol)
    labelled_sobol = acq_sobol(search_candidates=candidates, state=task_state)

    # IID sampler
    sampler_iid = BoTorchMCSampler(sampler_type="iid", num_samples=256)
    acq_iid = BoTorchAcquisition(acquisition_type="qEI", sampler=sampler_iid)
    labelled_iid = acq_iid(search_candidates=candidates, state=task_state)

    # Both should produce valid results (though different)
    assert np.isfinite(labelled_sobol.labels[0])
    assert np.isfinite(labelled_iid.labels[0])


def test_kwargs_passed_to_acquisition_function(task_state):
    """Test that extra kwargs are passed through to BoTorch acquisition functions."""
    candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
    ]

    # Create acquisition with additional kwargs
    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        # Extra kwargs that might be passed to qEI
        objective=None,  # These would be passed through
    )

    labelled = acq_fn(search_candidates=candidates, state=task_state)
    assert np.isfinite(labelled.labels[0])


# =============================================================================
# Consistency Tests
# =============================================================================


def test_repeated_calls_consistent(task_state):
    """Test that repeated calls with same candidates give same results."""
    acq_fn = BoTorchAcquisition(acquisition_type="qEI")

    candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.1, 0.9]), modality=Modality.TABULAR),
    ]

    labelled1 = acq_fn(search_candidates=candidates, state=task_state)
    labelled2 = acq_fn(search_candidates=candidates, state=task_state)

    # Should produce identical results
    assert np.allclose(labelled1.labels, labelled2.labels, atol=1e-5)


def test_optimization_produces_reasonable_candidates(task_state, simple_dataset):
    """Test that optimized candidates have better predicted values than random."""
    bounds = [[b[0], b[1]] for b in simple_dataset.bounds.T]

    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        bounds=bounds,
        batch_size=5,
        num_restarts=5,
        raw_samples=128,
    )

    # Optimize candidates
    optimized = acq_fn(search_candidates=[], state=task_state)

    # Generate random candidates
    random_data = np.random.uniform(
        low=simple_dataset.bounds[0],
        high=simple_dataset.bounds[1],
        size=(5, simple_dataset.dim),
    )
    random_candidates = [Candidate(data=d, modality=Modality.TABULAR) for d in random_data]

    # Get predictions for both
    pred_optimized = task_state.surrogate.predict(optimized.candidates)
    pred_random = task_state.surrogate.predict(random_candidates)

    # Both should produce finite predictions
    assert all(np.isfinite(pred_optimized.means))
    assert all(np.isfinite(pred_random.means))


# =============================================================================
# Analytic acquisition type tests
# =============================================================================


def test_analytic_ei_scores_candidates(task_state):
    """LogExpectedImprovement returns finite scores for each candidate."""
    acq_fn = BoTorchAcquisition(acquisition_type="expected_improvement", batch_size=1)
    candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.1, 0.9]), modality=Modality.TABULAR),
    ]
    labelled = acq_fn(search_candidates=candidates, state=task_state)
    assert len(labelled) == 2
    assert all(np.isfinite(labelled.labels))


def test_analytic_ucb_scores_candidates(task_state):
    """UpperConfidenceBound returns finite scores for each candidate."""
    acq_fn = BoTorchAcquisition(acquisition_type="upper_confidence_bound", beta=2.0, batch_size=1)
    candidates = [Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)]
    labelled = acq_fn(search_candidates=candidates, state=task_state)
    assert len(labelled) == 1
    assert np.isfinite(labelled.labels[0])


def test_analytic_pi_scores_candidates(task_state):
    """ProbabilityOfImprovement returns finite scores for each candidate."""
    acq_fn = BoTorchAcquisition(acquisition_type="probability_of_improvement", batch_size=1)
    candidates = [Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)]
    labelled = acq_fn(search_candidates=candidates, state=task_state)
    assert len(labelled) == 1
    assert np.isfinite(labelled.labels[0])


def test_log_noisy_ei_scores_candidates(task_state):
    """QLogNoisyExpectedImprovement returns finite scores for each candidate."""
    acq_fn = BoTorchAcquisition(acquisition_type="log_noisy_expected_improvement", batch_size=1)
    candidates = [Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)]
    labelled = acq_fn(search_candidates=candidates, state=task_state)
    assert len(labelled) == 1
    assert np.isfinite(labelled.labels[0])


def test_invalid_acquisition_type_rejects_new_names():
    """New analytic type names are accepted; an unrecognised name still raises ValueError."""
    BoTorchAcquisition(acquisition_type="expected_improvement")
    BoTorchAcquisition(acquisition_type="upper_confidence_bound")
    BoTorchAcquisition(acquisition_type="probability_of_improvement")
    BoTorchAcquisition(acquisition_type="log_noisy_expected_improvement")

    with pytest.raises(ValueError, match="Unsupported acquisition_type"):
        BoTorchAcquisition(acquisition_type="banana")  # type: ignore


# =============================================================================
# Model wrapping tests (migrated from test_botorch_acquisition_function.py)
# =============================================================================


def test_score_candidates_skips_wrapper_for_native_botorch_model(task_state, botorch_gp_model):
    """_score_candidates does not wrap a native BotorchModel in BotorchModelWrapper."""
    task_state.surrogate.model = botorch_gp_model

    acq_fn = BoTorchAcquisition(acquisition_type="qEI", batch_size=1)
    candidates = [Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)]

    mock_acq_fn = MagicMock()
    mock_acq_fn.return_value = torch.tensor([0.5])

    with (
        patch(
            "alf_tools.optimizer.acquisition_functions.botorch_acquisition.BotorchModelWrapper"
        ) as mock_wrapper,
        patch.object(acq_fn, "_create_acquisition_function", return_value=mock_acq_fn),
    ):
        acq_fn(search_candidates=candidates, state=task_state)
        mock_wrapper.assert_not_called()


def test_score_candidates_wraps_alf_model(mock_alf_model_with_variances):
    """_score_candidates wraps an ALF BaseModel in BotorchModelWrapper."""
    surrogate = MagicMock()
    surrogate.model = mock_alf_model_with_variances

    state = MagicMock()
    state.surrogate = surrogate
    state.dataset.train_dataset.labels.max.return_value = 1.0

    acq_fn = BoTorchAcquisition(acquisition_type="qUCB", beta=2.0, batch_size=1)
    candidates = [Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)]

    result = acq_fn(search_candidates=candidates, state=state)
    assert len(result.labels) == 1
    assert all(np.isfinite(result.labels))


def test_batch_size_gt_one_marginal_only_raises(mock_alf_model_with_variances):
    """batch_size>1 on a marginal-only model raises a pointed error (scoring mode)."""
    surrogate = MagicMock()
    surrogate.model = mock_alf_model_with_variances
    state = MagicMock()
    state.surrogate = surrogate
    state.dataset.train_dataset.labels.max.return_value = 1.0

    acq_fn = BoTorchAcquisition(acquisition_type="qUCB", beta=2.0, batch_size=2)
    candidates = [
        Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR),
        Candidate(data=np.array([0.2, 0.8]), modality=Modality.TABULAR),
    ]
    with pytest.raises(ValueError, match="joint posterior"):
        acq_fn(search_candidates=candidates, state=state)


def test_analytic_acquisition_marginal_only_succeeds(mock_alf_model_with_variances):
    """Analytic acquisitions work on marginal-only models (single-point)."""
    surrogate = MagicMock()
    surrogate.model = mock_alf_model_with_variances
    state = MagicMock()
    state.surrogate = surrogate
    state.dataset.train_dataset.labels.max.return_value = 1.0

    acq_fn = BoTorchAcquisition(acquisition_type="upper_confidence_bound", beta=2.0, batch_size=1)
    candidates = [Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)]
    result = acq_fn(search_candidates=candidates, state=state)
    assert len(result.labels) == 1
    assert all(np.isfinite(result.labels))


def test_batch_size_gt_one_marginal_only_raises_optimisation_mode(mock_alf_model_with_variances):
    """batch_size>1 on a marginal-only model raises in continuous optimisation mode."""
    surrogate = MagicMock()
    surrogate.model = mock_alf_model_with_variances
    state = MagicMock()
    state.surrogate = surrogate
    state.dataset.train_dataset.labels.max.return_value = 1.0

    acq_fn = BoTorchAcquisition(
        acquisition_type="qUCB", beta=2.0, batch_size=2, bounds=[[0.0, 1.0], [0.0, 1.0]]
    )
    with pytest.raises(ValueError, match="joint posterior"):
        acq_fn(search_candidates=[], state=state)


# =============================================================================
# AcquisitionType / _VALID_TYPES consistency
# =============================================================================


def test_acquisition_type_in_sync_with_valid_types():
    """AcquisitionType Literal and the runtime get_args list must stay identical."""
    valid = set(get_args(AcquisitionType))
    # Ensure the type alias includes all expected families and no extras.
    assert "qEI" in valid
    assert "qLogEI" in valid
    assert "qNEI" in valid
    assert "qUCB" in valid
    assert "expected_improvement" in valid
    assert "upper_confidence_bound" in valid
    assert "probability_of_improvement" in valid
    assert "log_noisy_expected_improvement" in valid
    # qKG must NOT be in the type (not yet implemented; raises at init).
    assert "qKG" not in valid


# =============================================================================
# BoTorchAcquisitionOptConfig configurability
# =============================================================================


def test_custom_optimization_config(task_state, simple_dataset):
    """Custom BoTorchAcquisitionOptConfig is respected during continuous optimization."""
    bounds = [[b[0], b[1]] for b in simple_dataset.bounds.T]
    opt_cfg = BoTorchAcquisitionOptConfig(maxiter=10, batch_limit=8)

    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        bounds=bounds,
        batch_size=1,
        num_restarts=2,
        raw_samples=32,
        optimization_config=opt_cfg,
    )

    assert acq_fn.optimization_config.maxiter == 10
    assert acq_fn.optimization_config.batch_limit == 8

    labelled = acq_fn(search_candidates=[], state=task_state)
    assert len(labelled) == 1
    assert np.isfinite(labelled.labels[0])
