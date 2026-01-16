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

import numpy as np
import pandas as pd
import pytest
import torch
from alf_core import (
    BaseDatasetConfig,
    FileTaskStateLogger,
    Oracle,
    SupervisedTask,
    Surrogate,
    TerminalTaskStateLogger,
)
from alf_tools.datasets import GFP
from alf_tools.models import GPModelConfig, GPModelTrainer, GPTrainConfig


@pytest.fixture
def set_seed():
    """Fixture to set random seeds for reproducible tests.

    Returns:
        A function that sets the random seeds.
    """

    def _set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return _set_seed


@pytest.fixture
def gfp_dataset():
    """Fixture to create a GFP dataset for testing.

    Returns:
        A GFP dataset.
    """
    config = BaseDatasetConfig(
        name="gfp",
        modality="sequence",
        seed=51505,
        train_ratio=0.1,
        test_ratio=0.2,
        validation_frac=0.5,
        split_type="random",
    )
    return GFP(config)


@pytest.fixture
def surrogate_model():
    """Fixture to create a GP surrogate model for testing.

    Returns:
        A surrogate model with a GP.
    """
    model_config = GPModelConfig(kernel_type="rbf", ard=False)
    train_config = GPTrainConfig(num_iterations=50, learning_rate=0.1)
    return Surrogate(model=GPModelTrainer(model_config=model_config, train_config=train_config))


@pytest.fixture
def oracle(gfp_dataset):
    """Fixture to create optimizer components.

    Returns:
        An Oracle.
    """
    oracle = Oracle(scorer=gfp_dataset)
    return oracle


class TestSupervisedGP:
    """Tests a supervised experiment with a GP surrogate model on the GFP dataset."""

    def test_supervised_gfp_gp_experiment(self, set_seed, gfp_dataset, surrogate_model, tmp_path):
        """Test the complete supervised GFP GP experiment pipeline.

        This test verifies that the supervised learning pipeline works correctly
        with the GP surrogate model. Unlike the CNN test, we don't check exact
        metrics since GP training can have more variability.
        """
        # Set seed for reproducible results
        set_seed(42)

        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "supervised_gfp_gp"
        save_path.mkdir()

        metrics_logger = TerminalTaskStateLogger()
        file_logger = FileTaskStateLogger(output_path=save_path)
        task_state_loggers = [metrics_logger, file_logger]

        # Create and run the supervised task
        task = SupervisedTask()
        state = task.setup(dataset=gfp_dataset, surrogate=surrogate_model)
        task.run(
            state=state,
            task_state_loggers=task_state_loggers,
        )

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Test that key metrics exist and are reasonable
        self._assert_metrics_exist_and_reasonable(metrics)

    def _assert_metrics_exist_and_reasonable(self, metrics: pd.DataFrame):
        """Assert that metrics exist and have reasonable values."""
        # Check surrogate metrics exist
        surrogate_metrics = [
            "surrogate/test_mse",
            "surrogate/test_spearman",
            "surrogate/test_pearson",
            "surrogate/test_pairwise_xent",
        ]
        for metric in surrogate_metrics:
            assert metric in metrics.columns, f"Metric {metric} should exist"
            value = metrics[metric].iloc[0]
            assert np.isfinite(value), f"Metric {metric} should be finite"

        # Check dataset metrics exist
        dataset_metrics = [
            "dataset/num_train",
            "dataset/train_mean",
            "dataset/num_validation",
            "dataset/validation_mean",
            "dataset/num_test",
            "dataset/test_mean",
            "dataset/num_candidate_pool",
        ]
        for metric in dataset_metrics:
            assert metric in metrics.columns, f"Metric {metric} should exist"
            value = metrics[metric].iloc[0]
            assert np.isfinite(value), f"Metric {metric} should be finite"

        # Check that Spearman correlation is in reasonable range [-1, 1]
        spearman = metrics["surrogate/test_spearman"].iloc[0]
        assert -1 <= spearman <= 1, f"Spearman should be in [-1, 1], got {spearman}"

        # Check that MSE is positive
        mse = metrics["surrogate/test_mse"].iloc[0]
        assert mse > 0, f"MSE should be positive, got {mse}"
