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

import numpy as np
import pandas as pd
import pytest
import torch
from alf_core import (
    BaseDatasetConfig,
    DatasetSearch,
    DesignTask,
    FileStateLogger,
    Optimizer,
    Oracle,
    Surrogate,
    TerminalStateLogger,
)
from alf_core.dataclasses.candidate import Candidate, Modality
from alf_core.dataclasses.labelled_candidates import LabelledCandidates
from alf_core.dataclasses.predictions import Predictions
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.model.base_model import BaseModel
from alf_tools.models import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.optimizer.acquisition_functions.ucb import UCB
from scipy.stats import spearmanr

# Expected warnings from running a small experiment end-to-end:
# - the 25-item test split is below the default num_acquisitions (100), so the
#   regret metrics fall back with a warning during evaluation;
# - the oracle re-labels acquired candidates with its own noise, so acquired
#   labels can exceed the raw dataset's best label and auc_top_k warns that
#   the normalised AUC is clamped.
pytestmark = [
    pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning"),
    pytest.mark.filterwarnings("ignore:Some round_values exceed best_value:UserWarning"),
]

# ---------------------------------------------------------------------------
# Synthetic sinusoidal helpers
# ---------------------------------------------------------------------------


def _sinusoidal_fn(x: np.ndarray) -> np.ndarray:
    """Target function: sin(x) + sin(2x) — deterministic, no noise.

    Returns:
        Array of function values.
    """
    return np.sin(x) + np.sin(2 * x)


class SinusoidalOracle(BaseModel):
    """Scores candidates by evaluating sin(x)+sin(2x) with small fixed-seed noise."""

    def __init__(self, noise_std: float = 0.1, seed: int = 0):
        """Initialize oracle with noise standard deviation and random seed.

        Args:
            noise_std: Standard deviation of additive Gaussian noise.
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)  # noqa: NPY002
        self.noise_std = noise_std

    def featurise(self, inputs: LabelledCandidates | list[Candidate]):
        """No-op featuriser — oracle evaluates analytically.

        Args:
            inputs: Input data to featurize (unused).
        """

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates) -> None:
        """No-op train — oracle is a closed-form function.

        Args:
            train_data: Training data (unused).
            val_data: Validation data (unused).
        """

    def predict(self, candidates: list[Candidate]) -> Predictions:
        """Predict noisy sinusoidal values for candidates.

        Args:
            candidates: List of TABULAR candidates with scalar data.

        Returns:
            Predictions with noisy sinusoidal means.
        """
        x = np.array([c.data.flatten()[0] for c in candidates])
        y = _sinusoidal_fn(x) + self.rng.randn(len(x)) * self.noise_std  # noqa: NPY002
        return Predictions(means=y)

    def sample(self, *args, **kwargs):
        """Not implemented for this oracle."""
        raise NotImplementedError

    def get_training_summary_metrics(self):
        """Return empty metrics dict — oracle has no training.

        Returns:
            Empty dictionary.
        """
        return {}


class SinusoidalDataset(BaseDataset):
    """Pre-generated sinusoidal dataset for offline e2e testing.

    Uses 250 evenly-spaced points on [0, 2π].
    With train_ratio=0.4, validation_frac=0.0, test_ratio=0.1 this gives:
      train=100, val=0, test=25, candidate_pool=125.
    """

    def load_dataset(self) -> LabelledCandidates:
        """Load sinusoidal dataset with 250 points on [0, 2π].

        Returns:
            LabelledCandidates with noisy sinusoidal observations.
        """
        rng = np.random.RandomState(42)  # noqa: NPY002
        x = np.linspace(0, 2 * np.pi, 250)
        y = _sinusoidal_fn(x) + rng.randn(250) * 0.1  # noqa: NPY002
        candidates = [Candidate(data=np.array([xi]), modality=Modality.TABULAR) for xi in x]
        return LabelledCandidates(candidates, y)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sinusoidal_dataset():
    """Sinusoidal dataset with 100 train / 0 val / 25 test / ~125 candidate_pool.

    validation_frac=0.0 ensures all acquired candidates go to train each round,
    so dataset/num_train grows by exactly acq_batch_size=5 per round.

    Returns:
        Configured SinusoidalDataset (not yet split).
    """
    config = BaseDatasetConfig(
        name="sinusoidal",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.4,
        validation_frac=0.0,
        test_ratio=0.1,
        split_type="random",
        problem_type="regression",
    )
    return SinusoidalDataset(config)


@pytest.fixture
def gp_surrogate():
    """GPModel with RBF kernel and scalar tabular featurizer.

    Returns:
        Surrogate wrapping a GPModel.
    """

    def tabular_featurizer(seqs):
        # seqs is a list of np.arrays of shape (1,) from Candidate.data
        return torch.tensor(np.array(seqs), dtype=torch.float32)  # (n, 1)

    model_config = GPModelConfig(kernel_type="rbf", ard=False, mean_type="constant")
    train_config = GPTrainConfig(
        learning_rate=0.1,
        num_iterations=50,
        optimizer_type="adam",
        log_frequency=50,
    )
    featurizer_config = FeaturizerConfig(
        featurizer_type="custom",
        custom_featurizer=tabular_featurizer,
    )
    gp = GPModel(
        name="gp_sinusoidal_e2e",
        model_config=model_config,
        train_config=train_config,
        featurizer_config=featurizer_config,
        device="cpu",
    )
    return Surrogate(model=gp)


@pytest.fixture
def ucb_optimizer():
    """UCB acquisition + DatasetSearch optimizer.

    Returns:
        Optimizer with UCB acquisition and dataset search.
    """
    return Optimizer(
        acquisition_fn=UCB(alpha=0.9),
        search_fn=DatasetSearch(),
    )


@pytest.fixture
def sinusoidal_oracle():
    """Sinusoidal oracle (fixed seed for reproducibility).

    Note: We use a separate SinusoidalOracle(BaseModel) as the scorer to
    keep dataset and scoring concerns separate. This avoids requiring the
    dataset to implement BaseModel.predict().

    Returns:
        Oracle wrapping SinusoidalOracle.
    """
    return Oracle(scorer=SinusoidalOracle(noise_std=0.1, seed=0))


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestDesignGPSinusoidalSurrogate:
    """End-to-end test for GP surrogate in a DesignTask on sinusoidal data."""

    def test_design_gp_sinusoidal_experiment(
        self,
        sinusoidal_dataset,
        gp_surrogate,
        ucb_optimizer,
        sinusoidal_oracle,
        tmp_path,
    ):
        """Runs a 3-round DesignTask with GPModel surrogate and verifies outputs.

        Structural assertions (strict):
        - metrics.csv is created with exactly 4 rows (round 0 + 3 rounds)
        - Required columns are present (including an error metric)
        - dataset/num_train grows by exactly acq_batch_size each round

        Behavioral assertions (loose tolerance):
        - direct Spearman on test set > 0.4 (GP should learn sinusoid)
        - direct MSE on test set is finite and positive
        - surrogate/test_residual_spearman values are finite
        - all acquired_candidates/round_mean values are finite
        - post-training predictions have non-negative variances
        - learned hyperparameters (noise, outputscale, lengthscale) are positive
        """
        save_path = tmp_path / "gp_sinusoidal_e2e"
        save_path.mkdir()

        state_loggers = [
            TerminalStateLogger(),
            FileStateLogger(output_path=save_path),
        ]

        sinusoidal_dataset.setup()  # BaseTask.setup() does not call dataset.setup()
        task = DesignTask(num_acq_rounds=3, acq_batch_size=5)
        state = task.setup(dataset=sinusoidal_dataset, surrogate=gp_surrogate)
        initial_num_train = len(state.dataset.train_dataset)

        task.run(
            state=state,
            state_loggers=state_loggers,
            optimizer=ucb_optimizer,
            oracle=sinusoidal_oracle,
        )

        # --- Structural assertions ---
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "metrics.csv must be created"

        metrics = pd.read_csv(metrics_file)

        # 4 rows: round 0 + 3 acquisition rounds. Summary goes to summary.csv.
        assert len(metrics) == 4, f"Expected 4 rows, got {len(metrics)}"

        # Required columns — surrogate/test_ece is the error metric present
        # in the CSV when the surrogate produces variances (GP always does).
        # mse/spearman are only emitted when variances are absent.
        required_cols = [
            "dataset/num_train",
            "surrogate/test_ece",
            "surrogate/test_residual_spearman",
            "acquired_candidates/round_mean",
        ]
        for col in required_cols:
            assert col in metrics.columns, f"Missing column: {col}"

        # dataset/num_train grows by exactly acq_batch_size each round because
        # validation_frac=0.0 — all acquired candidates go to train.
        acq_batch_size = 5
        expected_num_train = [initial_num_train + i * acq_batch_size for i in range(4)]
        actual_num_train = metrics["dataset/num_train"].dropna().astype(int).tolist()
        assert actual_num_train == expected_num_train, (
            f"Expected num_train={expected_num_train}, got {actual_num_train}"
        )

        # --- Behavioral assertions (loose) ---

        # residual_spearman measures calibration (residuals vs variances).
        # Assert values are finite real numbers.
        residual_spearman_vals = metrics["surrogate/test_residual_spearman"].dropna()
        assert np.all(np.isfinite(residual_spearman_vals)), (
            "residual_spearman values must be finite"
        )

        # Verify GP prediction quality directly via Spearman on the test set.
        gp_model_check = gp_surrogate.model
        test_x = np.linspace(0, 2 * np.pi, 25)
        test_cands = [Candidate(data=np.array([x]), modality=Modality.TABULAR) for x in test_x]
        test_preds = gp_model_check.predict(test_cands)
        test_targets = _sinusoidal_fn(test_x)
        spearman_val = spearmanr(test_targets, test_preds.means)[0]
        assert spearman_val > 0.4, (
            f"Direct Spearman {spearman_val:.3f} < 0.4; "
            "GP may not have learned the sinusoidal function"
        )

        # MSE should be finite and positive.
        # surrogate/test_mse is not emitted when variances are present
        # (the metrics registry only runs variance-requiring metrics for GP).
        # We compute MSE directly from the GP predictions made above.
        mse_value = float(np.mean((test_targets - test_preds.means) ** 2))
        assert np.isfinite(mse_value), "Direct test MSE must be finite"
        assert mse_value > 0, "Direct test MSE must be positive"

        # Acquired candidate means should be finite
        round_means = metrics["acquired_candidates/round_mean"].dropna()
        assert np.all(np.isfinite(round_means)), "All round_mean values must be finite"

        # Post-training predictions must have non-negative variances
        var_cands = [
            Candidate(data=np.array([x]), modality=Modality.TABULAR)
            for x in np.linspace(0, 2 * np.pi, 20)
        ]
        preds = gp_model_check.predict(var_cands)
        assert np.all(preds.variances >= 0), "GP variances must be non-negative"
        assert np.all(np.isfinite(preds.means)), "GP means must be finite"

        # Learned hyperparameters must be positive
        hyperparams = gp_model_check.get_hyperparameters()
        assert hyperparams["noise"] > 0, "Noise hyperparameter must be positive"
        assert hyperparams["outputscale"] > 0, "Outputscale hyperparameter must be positive"
        lengthscale = hyperparams["lengthscale"]
        if isinstance(lengthscale, np.ndarray):
            assert np.all(lengthscale > 0), "All lengthscale values must be positive"
        else:
            assert lengthscale > 0, "Lengthscale hyperparameter must be positive"
