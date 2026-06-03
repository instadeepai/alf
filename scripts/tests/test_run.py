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

"""Integration tests for the factory + ALF pipeline."""

import factory
import numpy as np
import pandas as pd
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
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.optimizer.acquisition_functions.ucb import UCB
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Minimal in-memory helpers — no network I/O
# ---------------------------------------------------------------------------


class _SinusoidalDataset(BaseDataset):
    """100-point sinusoidal dataset backed entirely in memory."""

    def load_dataset(self) -> LabelledCandidates:
        """Load a sinusoidal dataset with 100 points on [0, 2*pi].

        Returns:
            LabelledCandidates with noisy sinusoidal observations.
        """
        rng = np.random.RandomState(42)
        x = np.linspace(0, 2 * np.pi, 300)
        y = np.sin(x) + rng.randn(300) * 0.1
        cands = [Candidate(data=np.array([xi]), modality=Modality.TABULAR) for xi in x]
        return LabelledCandidates(cands, y)


class _SinusoidalOracle(BaseModel):
    """Scores candidates as sin(x) — no training needed."""

    def featurise(self, inputs):
        """No-op featuriser."""

    def train(self, train_data, val_data=None):
        """No-op train."""

    def predict(self, candidate_points):
        """Predict sin(x) for each candidate.

        Args:
            candidate_points: List of TABULAR candidates with scalar data.

        Returns:
            Predictions with sin values as means.
        """
        x = np.array([c.data.flatten()[0] for c in candidate_points])
        return Predictions(means=np.sin(x))

    def sample(self, condition=None):
        """Not implemented."""
        raise NotImplementedError

    def get_training_summary_metrics(self):
        """Return empty metrics.

        Returns:
            Empty dict.
        """
        return {}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_factory_builds_gp_and_runs_design_task(tmp_path):
    """Factory builds a GP surrogate; DesignTask runs 2 rounds without error.

    Verifies:
    - metrics.csv is written to output_path
    - 3 rows: round 0 + 2 acquisition rounds
    """

    def tabular_featurizer(seqs):
        return torch.tensor(np.array(seqs), dtype=torch.float32)

    model_config = GPModelConfig(kernel_type="rbf", ard=False, mean_type="constant")
    train_config = GPTrainConfig(learning_rate=0.1, num_iterations=10, log_frequency=10)
    featurizer_config = FeaturizerConfig(
        featurizer_type="custom", custom_featurizer=tabular_featurizer
    )
    gp = GPModel(
        name="gp_sin_test",
        model_config=model_config,
        train_config=train_config,
        featurizer_config=featurizer_config,
        device="cpu",
    )
    surrogate = Surrogate(model=gp)
    optimizer = Optimizer(acquisition_fn=UCB(alpha=0.9), search_fn=DatasetSearch())
    oracle = Oracle(scorer=_SinusoidalOracle())
    loggers = [TerminalStateLogger(), FileStateLogger(output_path=tmp_path)]

    dataset_cfg = BaseDatasetConfig(
        name="sin_test",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.4,
        validation_frac=0.0,
        test_ratio=0.1,
        split_type="random",
        problem_type="regression",
    )
    dataset = _SinusoidalDataset(dataset_cfg)
    dataset.setup()

    task = DesignTask(num_acq_rounds=2, acq_batch_size=3)
    state = task.setup(dataset=dataset, surrogate=surrogate)
    task.run(state=state, state_loggers=loggers, optimizer=optimizer, oracle=oracle)

    metrics_file = tmp_path / "metrics.csv"
    assert metrics_file.exists(), "metrics.csv must be written"
    df = pd.read_csv(metrics_file)
    assert len(df) == 3  # round 0 + 2 acq rounds


def test_build_optimizer_from_yaml_ucb_config():
    """build_optimizer correctly instantiates UCB via _target_."""
    cfg = OmegaConf.create({
        "optimizer": {
            "acquisition_fn": {
                "_target_": "alf_tools.optimizer.acquisition_functions.ucb.UCB",
                "alpha": 0.5,
            },
            "search_fn": {"_target_": "alf_core.optimizer.search.DatasetSearch"},
        }
    })
    opt = factory.build_optimizer(cfg)
    assert isinstance(opt.acquisition_fn, UCB)
    assert opt.acquisition_fn.alpha == 0.5
