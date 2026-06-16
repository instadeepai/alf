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

import logging
import math

import gpytorch
import numpy as np
import pytest
import torch
from alf_core import Candidate, LabelledCandidates
from alf_core.dataclasses.candidate import Modality
from alf_tools.models.gp import (
    FeaturizerConfig,
    GPModel,
    GPModelConfig,
    GPTrainConfig,
)
from alf_tools.models.utils import build_from_target, one_hot_encode
from botorch.models import SingleTaskGP

# Training on deliberately tiny datasets triggers the regret-metric fallback
# warning (num_acquisitions > available items) from alf_core during the
# train/validation metric computation in GPModel.train(), and can produce
# constant predictions or targets that make scipy's correlation metrics warn.
pytestmark = pytest.mark.filterwarnings(
    "ignore:num_acquisitions:UserWarning",
    "ignore::scipy.stats.ConstantInputWarning",
    "ignore::scipy.stats.NearConstantInputWarning",
)


@pytest.fixture
def sample_sinusoidal_data():
    """Sample data from a sinusoidal function.

    Returns:
        LabelledCandidates with sinusoidal training data.
    """
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.randn(100) * 0.1
    candidates = [Candidate(data=x, modality="tabular") for x in x]
    labels = np.array(y)
    return LabelledCandidates(candidates, labels)


@pytest.fixture
def val_sinusoidal_data():
    """Sample data from a sinusoidal function.

    Returns:
        LabelledCandidates with sinusoidal validation data.
    """
    x = np.linspace(1, 10, 10)
    y = np.sin(x) + np.random.randn(10) * 0.1
    candidates = [Candidate(data=x, modality="tabular") for x in x]
    labels = np.array(y)
    return LabelledCandidates(candidates, labels)


@pytest.fixture
def sample_data():
    """Create sample training data.

    Returns:
        A LabelledCandidates object containing the training data.
    """
    sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 10  # Simple repeated sequence
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(10) * 0.5 + 1.0
    return LabelledCandidates(candidates, labels)


@pytest.fixture
def sample_val_data():
    """Create sample validation data.

    Returns:
        A LabelledCandidates object containing the validation data.
    """
    sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 2  # Simple repeated sequence
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(2) * 0.5 + 1.0
    return LabelledCandidates(candidates, labels)


@pytest.fixture
def gp_model():
    """Create a GPModel with small settings for fast testing.

    Returns:
        A GPModel.
    """
    model_config = GPModelConfig(kernel_type="rbf", ard=False)
    train_config = GPTrainConfig(num_iterations=200, log_frequency=5)
    featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
    return GPModel(
        name="test_gp",
        model_config=model_config,
        train_config=train_config,
        featurizer_config=featurizer_config,
        device="cpu",
    )


@pytest.fixture
def gp_model_sinusoidal():
    """Create a GPModel with small settings for fast testing.
    Uses a custom featurizer that converts the input to a tensor.

    Returns:
        A GPModel.
    """
    model_config = GPModelConfig(kernel_type="rbf", ard=False)
    train_config = GPTrainConfig(num_iterations=10, log_frequency=5)
    featurizer_config = FeaturizerConfig(
        featurizer_type="custom",
        custom_featurizer=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
    )
    return GPModel(
        name="test_gp_sinusoidal",
        model_config=model_config,
        train_config=train_config,
        featurizer_config=featurizer_config,
        device="cpu",
    )


class TestGPModel:
    """Test the GPModel class."""

    def test_train_and_predict(self, gp_model, sample_data, sample_val_data):
        """Test the full training and prediction pipeline."""
        # Train should work without error
        gp_model.train(sample_data, sample_val_data)

        # Model should be initialized
        assert gp_model.gp_model is not None
        assert gp_model.likelihood is not None

        # Should store metrics
        metrics = gp_model.get_training_summary_metrics()
        assert "final_mll" in metrics
        assert "final_train_residual_pearson" in metrics

        # Predictions should work and include variances
        test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * 3
        predictions = gp_model.predict(test_candidates)
        assert predictions.means.shape == (3,)
        assert predictions.variances.shape == (3,)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(np.isfinite(predictions.variances))
        # Variances must be non-negative
        assert np.all(predictions.variances >= 0)

    def test_train_with_validation(self, gp_model, sample_data):
        """Test training with validation data."""
        val_sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 3
        val_candidates = [Candidate(data=seq, modality="sequence") for seq in val_sequences]
        val_data = LabelledCandidates(val_candidates, np.random.randn(3))

        gp_model.train(sample_data, val_data=val_data)

        metrics = gp_model.get_training_summary_metrics()
        assert "final_mll" in metrics

    def test_sample_raises_not_implemented_error(self, gp_model):
        """sample() raises NotImplementedError for the discriminative GP."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            gp_model.sample()

    def test_predict_before_train_raises_error(self, gp_model):
        """Test that predicting before train raises an error."""
        candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        with pytest.raises(RuntimeError, match="Model not trained"):
            gp_model.predict(candidates)

    def test_one_hot_encoding(self, gp_model):
        """Test that one-hot encoding produces correct shape and values."""
        sequences = ["ACDE", "FGHI"]
        encoded = one_hot_encode(
            sequences, gp_model.char_to_idx, gp_model.alphabet_size, flatten=True
        )

        # Check shape: (batch_size, alphabet_size * seq_length) when flattened
        assert encoded.shape == (2, 20 * 4)
        # Check that encoding is binary
        assert torch.all((encoded == 0) | (encoded == 1))

    def test_one_hot_encoding_not_flattened(self):
        """Test one-hot encoding without flattening."""
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=False)
        model_config = GPModelConfig(kernel_type="rbf")
        train_config = GPTrainConfig(num_iterations=5)
        gp_model = GPModel(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        sequences = ["ACDE", "FGHI"]
        encoded = one_hot_encode(
            sequences, gp_model.char_to_idx, gp_model.alphabet_size, flatten=False
        )

        # Check shape: (batch_size, alphabet_size, seq_length) when not flattened
        assert encoded.shape == (2, 20, 4)

    def test_get_hyperparameters(self, gp_model, sample_data, sample_val_data):
        """Test that we can extract learned hyperparameters."""
        gp_model.train(sample_data, val_data=sample_val_data)
        hyperparams = gp_model.get_hyperparameters()

        assert "noise" in hyperparams
        assert "lengthscale" in hyperparams
        assert "outputscale" in hyperparams
        assert "mean_constant" in hyperparams

        # Check that hyperparameters are reasonable
        assert hyperparams["noise"] > 0
        assert hyperparams["outputscale"] > 0

    def test_different_kernels(self, sample_data, sample_val_data):
        """Test that different kernel types work."""
        kernel_types = ["rbf", "matern", "linear", "polynomial", "rbf_linear"]

        for kernel_type in kernel_types:
            model_config = GPModelConfig(kernel_type=kernel_type, ard=False)
            train_config = GPTrainConfig(num_iterations=5, log_frequency=10)
            featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
            gp_model = GPModel(
                model_config=model_config,
                train_config=train_config,
                featurizer_config=featurizer_config,
                device="cpu",
            )

            # Should train without error
            gp_model.train(sample_data, sample_val_data)

            # Should predict without error
            test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
            predictions = gp_model.predict(test_candidates)
            assert predictions.means.shape == (1,)
            assert predictions.variances.shape == (1,)

    def test_ard_kernel(self, sample_data, sample_val_data):
        """Test that ARD (Automatic Relevance Determination) works."""
        model_config = GPModelConfig(kernel_type="rbf", ard=True)
        train_config = GPTrainConfig(num_iterations=10)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
        gp_model = GPModel(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data, sample_val_data)
        hyperparams = gp_model.get_hyperparameters()

        # With ARD, lengthscale should be an array
        assert isinstance(hyperparams["lengthscale"], np.ndarray)
        # Should have multiple lengthscales
        assert len(hyperparams["lengthscale"]) > 1

    def test_custom_featurizer(self, sample_data, sample_val_data):
        """Test using a custom featurization function."""

        def custom_featurizer(sequences: list[str]) -> torch.Tensor:
            # Simple custom featurizer: length of sequence
            features = torch.tensor([[len(seq)] for seq in sequences], dtype=torch.float32)
            return features

        featurizer_config = FeaturizerConfig(
            featurizer_type="custom", custom_featurizer=custom_featurizer
        )
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(num_iterations=10)
        gp_model = GPModel(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data, sample_val_data)

        test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        predictions = gp_model.predict(test_candidates)
        assert predictions.means.shape == (1,)
        assert predictions.variances.shape == (1,)

    def test_reproducibility_with_seed(self, sample_data, sample_val_data):
        """Test that training is reproducible when using the same seed."""

        def set_seed(seed: int):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Create two identical models
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(num_iterations=10)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)

        set_seed(42)
        gp_model1 = GPModel(
            name="test_gp_1",
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )
        gp_model1.train(sample_data, sample_val_data)

        set_seed(42)
        gp_model2 = GPModel(
            name="test_gp_2",
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )
        gp_model2.train(sample_data, sample_val_data)

        # Get predictions from both models
        test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        pred1 = gp_model1.predict(test_candidates)
        pred2 = gp_model2.predict(test_candidates)

        # Predictions should be very close
        np.testing.assert_allclose(pred1.means, pred2.means, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(pred1.variances, pred2.variances, rtol=1e-5, atol=1e-7)

    def test_matern_kernel_smoothness(self, sample_data, sample_val_data):
        """Test Matern kernel with different smoothness parameters."""
        nu_values = [0.5, 1.5, 2.5]

        for nu in nu_values:
            model_config = GPModelConfig(kernel_type="matern", matern_nu=nu, ard=False)
            train_config = GPTrainConfig(num_iterations=5)
            featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
            gp_model = GPModel(
                model_config=model_config,
                train_config=train_config,
                featurizer_config=featurizer_config,
                device="cpu",
            )

            # Should train without error
            gp_model.train(sample_data, sample_val_data)

            # Should predict without error
            test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
            predictions = gp_model.predict(test_candidates)
            assert predictions.means.shape == (1,)

    def test_adam_optimizer(self, sample_data, sample_val_data):
        """Test training with Adam optimizer."""
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(optimizer_type="adam", learning_rate=0.1, num_iterations=10)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
        gp_model = GPModel(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data, sample_val_data)
        assert gp_model.gp_model is not None

    def test_lbfgs_optimizer(self, sample_data, sample_val_data):
        """Test training with L-BFGS optimizer."""
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(optimizer_type="lbfgs", learning_rate=0.1, num_iterations=5)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
        gp_model = GPModel(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data, sample_val_data)
        assert gp_model.gp_model is not None

    def test_sinusoidal_data(
        self, gp_model_sinusoidal, sample_sinusoidal_data, val_sinusoidal_data
    ):
        """Test the model to learn the sinusoidal function."""
        gp_model_sinusoidal.train(
            sample_sinusoidal_data,
            val_data=val_sinusoidal_data,
        )
        test_candidates = [Candidate(data=x, modality="tabular") for x in np.linspace(0, 10, 10)]
        predictions = gp_model_sinusoidal.predict(test_candidates)
        assert predictions.means.shape == (10,)
        assert predictions.variances.shape == (10,)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(np.isfinite(predictions.variances))
        # Variances must be non-negative
        assert np.all(predictions.variances >= 0)


class TestGPNormalisation:
    """Tests for GPModel input normalisation and output standardisation."""

    def test_gp_train_config_default_normalises_inputs(self) -> None:
        """GPTrainConfig must default normalise_inputs_strategy to 'minmax'."""
        config = GPTrainConfig()
        assert config.normalise_inputs_strategy == "minmax"

    def test_input_normalisation_enabled(self, sample_data, sample_val_data):
        """A 'minmax' strategy must run without error; predictions in original label scale."""
        model = GPModel(
            train_config=GPTrainConfig(num_iterations=5, normalise_inputs_strategy="minmax"),
            featurizer_config=FeaturizerConfig(featurizer_type="one_hot"),
            device="cpu",
        )
        model.train(sample_data, val_data=sample_val_data)
        assert model._input_transform is not None
        assert model._input_transform.is_fitted

        predictions = model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(predictions.variances >= 0)

    def test_input_normalisation_disabled(self, sample_data, sample_val_data):
        """A None strategy leaves _input_transform as None and predict still works."""
        model = GPModel(
            train_config=GPTrainConfig(num_iterations=5, normalise_inputs_strategy=None),
            featurizer_config=FeaturizerConfig(featurizer_type="one_hot"),
            device="cpu",
        )
        model.train(sample_data, val_data=sample_val_data)
        assert model._input_transform is None

        predictions = model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(predictions.variances >= 0)

    def test_output_standardisation_enabled(self, sample_data, sample_val_data):
        """standardise_outputs=True trains in standardised space; predict returns original scale."""
        model = GPModel(
            train_config=GPTrainConfig(num_iterations=5, standardise_outputs=True),
            featurizer_config=FeaturizerConfig(featurizer_type="one_hot"),
            device="cpu",
        )
        model.train(sample_data, val_data=sample_val_data)
        assert model._output_standardiser is not None
        assert model._output_standardiser.is_fitted

        predictions = model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(predictions.variances >= 0)
        # Predictions must be in original label scale, not standardised space
        label_scale = sample_data.labels.std()
        assert predictions.means.std() < label_scale * 10  # sanity: not blown up

    def test_standardise_outputs_train_and_val_metrics_are_inverse_transformed(self):
        """Both final_train and final_val summary metrics must be in original label scale.

        Labels have std=50. In standardised (Z-score) space, predicted CIs have width ≈ 4
        and cannot contain targets scattered over ±150, giving coverage ≈ 0.  After correct
        inverse-transform the CI is 50× wider and should contain most targets (coverage ≈ 0.95).
        A threshold of 0.3 cleanly separates these two cases for both train and val metrics.
        """
        rng = np.random.default_rng(0)
        n_train, n_val = 10, 10
        label_std = 50.0
        train_labels = rng.standard_normal(n_train) * label_std
        val_labels = rng.standard_normal(n_val) * label_std

        train_data = LabelledCandidates(
            [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * n_train,
            train_labels,
        )
        val_data = LabelledCandidates(
            [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * n_val,
            val_labels,
        )

        model = GPModel(
            train_config=GPTrainConfig(num_iterations=10, standardise_outputs=True),
            featurizer_config=FeaturizerConfig(featurizer_type="one_hot"),
            device="cpu",
        )
        model.train(train_data, val_data=val_data)
        metrics = model.get_training_summary_metrics()

        # coverage_0.95 = fraction of targets inside the 95% CI.
        # Correct (original scale): CI ≈ ±98, targets ~ N(0, 50) → coverage ≈ 0.95.
        # Bug (standardised scale): CI ≈ ±2, targets at ±50s → coverage ≈ 0.
        assert "final_train_coverage_0.95" in metrics
        assert "final_val_coverage_0.95" in metrics
        assert metrics["final_train_coverage_0.95"] > 0.3, (
            f"final_train_coverage_0.95={metrics['final_train_coverage_0.95']:.3f} — "
            "train metrics appear to be in standardised space, not original label scale"
        )
        assert metrics["final_val_coverage_0.95"] > 0.3, (
            f"final_val_coverage_0.95={metrics['final_val_coverage_0.95']:.3f} — "
            "val metrics appear to be in standardised space, not original label scale"
        )


class TestGPModelConfigDefaults:
    """Tests for the updated GPModelConfig dict-based fields."""

    def test_lengthscale_prior_defaults_to_lognormal_dict(self):
        """Test that the default lengthscale_prior is a dict specifying a
        LogNormalPrior with the expected parameters.
        """
        cfg = GPModelConfig()
        assert isinstance(cfg.lengthscale_prior, dict)
        assert cfg.lengthscale_prior["_target_"] == "gpytorch.priors.LogNormalPrior"
        assert cfg.lengthscale_prior["loc"] == pytest.approx(math.sqrt(2))
        assert cfg.lengthscale_prior["scale"] == pytest.approx(math.sqrt(3))

    def test_lengthscale_constraint_defaults_to_none(self):
        """Test that the default lengthscale_constraint is None, meaning no
        constraint is applied to the lengthscale parameter.
        """
        cfg = GPModelConfig()
        assert cfg.lengthscale_constraint is None

    def test_outputscale_prior_defaults_to_none(self):
        """Test that the default outputscale_prior is None, meaning no prior is
        applied to the outputscale parameter.
        """
        cfg = GPModelConfig()
        assert cfg.outputscale_prior is None

    def test_noise_constraint_defaults_to_none(self):
        """Test that the default noise_constraint is None, meaning no constraint
        is applied to the noise parameter.
        """
        cfg = GPModelConfig()
        assert cfg.noise_constraint is None

    def test_custom_gamma_prior_accepted(self):
        """Test that a custom GammaPrior configuration is accepted and stored
        correctly in the config.
        """
        prior_cfg = {
            "_target_": "gpytorch.priors.GammaPrior",
            "concentration": 3.0,
            "rate": 6.0,
        }
        cfg = GPModelConfig(lengthscale_prior=prior_cfg)
        assert cfg.lengthscale_prior["_target_"] == "gpytorch.priors.GammaPrior"

    def test_lengthscale_prior_can_be_set_to_none(self):
        """Test that setting lengthscale_prior to None is accepted and results
        in no prior being used.
        """
        cfg = GPModelConfig(lengthscale_prior=None)
        assert cfg.lengthscale_prior is None

    def test_build_from_target_works_with_default_prior(self):
        """Test that the default lengthscale_prior dict can be successfully
        built into a LogNormalPrior instance.
        """
        cfg = GPModelConfig()
        prior = build_from_target(cfg.lengthscale_prior)
        assert isinstance(prior, gpytorch.priors.LogNormalPrior)

    def test_default_prior_dicts_are_independent(self):
        """Each GPModelConfig() gets its own dict, not a shared reference."""
        cfg1 = GPModelConfig()
        cfg2 = GPModelConfig()
        assert cfg1.lengthscale_prior is not cfg2.lengthscale_prior


class TestGPModelPriorWiring:
    """Integration tests: confirm prior and constraint reach the kernel."""

    _train_cfg = GPTrainConfig(num_iterations=5, log_frequency=5)
    _feat_cfg = FeaturizerConfig(
        featurizer_type="custom",
        custom_featurizer=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
    )

    def test_default_lognormal_prior_is_registered(
        self, sample_sinusoidal_data, val_sinusoidal_data
    ):
        """Test that the default LogNormalPrior is correctly registered on the kernel."""
        model = GPModel(
            model_config=GPModelConfig(),
            train_config=self._train_cfg,
            featurizer_config=self._feat_cfg,
        )
        model.train(sample_sinusoidal_data, val_sinusoidal_data)
        base_kernel = model.gp_model.covar_module.base_kernel
        # Prior should be registered on the base kernel
        prior_names = {name for name, *_ in base_kernel.named_priors()}
        assert "lengthscale_prior" in prior_names

    def test_gamma_prior_is_registered(self, sample_sinusoidal_data, val_sinusoidal_data):
        """Test that a custom GammaPrior is correctly registered on the kernel."""
        cfg = GPModelConfig(
            lengthscale_prior={
                "_target_": "gpytorch.priors.GammaPrior",
                "concentration": 3.0,
                "rate": 6.0,
            }
        )
        model = GPModel(
            model_config=cfg,
            train_config=self._train_cfg,
            featurizer_config=self._feat_cfg,
        )
        model.train(sample_sinusoidal_data, val_sinusoidal_data)
        base_kernel = model.gp_model.covar_module.base_kernel
        named = {name: prior for name, _module, prior, *_ in base_kernel.named_priors()}
        assert "lengthscale_prior" in named
        assert isinstance(named["lengthscale_prior"], gpytorch.priors.GammaPrior)

    def test_no_prior_when_lengthscale_prior_is_none(
        self, sample_sinusoidal_data, val_sinusoidal_data
    ):
        """Test that setting lengthscale_prior to None results in no prior being registered."""
        cfg = GPModelConfig(lengthscale_prior=None)
        model = GPModel(
            model_config=cfg,
            train_config=self._train_cfg,
            featurizer_config=self._feat_cfg,
        )
        model.train(sample_sinusoidal_data, val_sinusoidal_data)
        base_kernel = model.gp_model.covar_module.base_kernel
        prior_names = {name for name, *_ in base_kernel.named_priors()}
        assert "lengthscale_prior" not in prior_names

    def test_lengthscale_constraint_is_applied(self, sample_sinusoidal_data, val_sinusoidal_data):
        """Test that a custom lengthscale constraint is correctly applied to the kernel."""
        cfg = GPModelConfig(
            lengthscale_constraint={
                "_target_": "gpytorch.constraints.GreaterThan",
                "lower_bound": 0.05,
            }
        )
        model = GPModel(
            model_config=cfg,
            train_config=self._train_cfg,
            featurizer_config=self._feat_cfg,
        )
        model.train(sample_sinusoidal_data, val_sinusoidal_data)
        base_kernel = model.gp_model.covar_module.base_kernel
        assert hasattr(base_kernel, "raw_lengthscale_constraint")
        assert isinstance(
            base_kernel.raw_lengthscale_constraint,
            gpytorch.constraints.GreaterThan,
        )

    def test_ard_warning_with_default_prior(self, tabular_data, caplog):
        """Test that enabling ARD with the default LogNormalPrior triggers
        a warning about Hvarfner priors.
        """
        model = GPModel(
            model_config=GPModelConfig(ard=True),
            train_config=self._train_cfg,
            featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
            device="cpu",
        )
        with caplog.at_level(logging.WARNING, logger="alf-tools"):
            model.train(tabular_data)
        assert "ARD is enabled" in caplog.text
        assert "Hvarfner" in caplog.text

    def test_ard_warning_with_no_prior(self, tabular_data, caplog):
        """Test that enabling ARD with lengthscale_prior=None triggers a
        warning about the missing prior.
        """
        model = GPModel(
            model_config=GPModelConfig(ard=True, lengthscale_prior=None),
            train_config=self._train_cfg,
            featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
            device="cpu",
        )
        with caplog.at_level(logging.WARNING, logger="alf-tools"):
            model.train(tabular_data)
        assert "ARD is enabled" in caplog.text
        assert "no lengthscale prior" in caplog.text

    def test_no_ard_warning_for_one_dimensional_inputs(self, sample_sinusoidal_data, caplog):
        """Test that no ARD warning is logged for 1-D inputs, where the
        default prior already matches the Hvarfner recommendation.
        """
        model = GPModel(
            model_config=GPModelConfig(ard=True),
            train_config=self._train_cfg,
            featurizer_config=self._feat_cfg,
        )
        with caplog.at_level(logging.WARNING, logger="alf-tools"):
            model.train(sample_sinusoidal_data)
        assert "ARD is enabled" not in caplog.text


@pytest.fixture
def tabular_data():
    """Create tabular training data with precomputed numpy features.

    Returns:
        LabelledCandidates with 3-dimensional tabular candidates.
    """
    rng = np.random.default_rng(0)
    X = rng.random((20, 3))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + rng.standard_normal(20) * 0.05
    candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]
    return LabelledCandidates(candidates, y)


@pytest.fixture
def precomputed_gp_model():
    """Create a GPModel using precomputed features for fast testing.

    Returns:
        A GPModel.
    """
    return GPModel(
        model_config=GPModelConfig(kernel_type="rbf", ard=False),
        train_config=GPTrainConfig(num_iterations=10),
        featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
        device="cpu",
    )


class TestGPBoTorchBackbone:
    """Tests for the BoTorch SingleTaskGP backbone of GPModel."""

    def test_scipy_optimizer_trains(self, tabular_data):
        """optimizer_type='scipy' trains and records strictly increasing epochs."""
        model = GPModel(
            model_config=GPModelConfig(kernel_type="rbf", ard=False),
            train_config=GPTrainConfig(optimizer_type="scipy", num_iterations=20),
            featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
            device="cpu",
        )
        model.train(tabular_data, None)
        assert model.gp_model is not None

        epoch_metrics = model.get_epoch_metrics()
        assert len(epoch_metrics) > 0
        epochs = [m.epoch for m in epoch_metrics]
        assert all(a < b for a, b in zip(epochs, epochs[1:]))

        metrics = model.get_training_summary_metrics()
        assert "final_mll" in metrics
        assert "final_loss" in metrics

    def test_invalid_optimizer_type_raises(self):
        """An unsupported optimizer_type raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unsupported optimizer_type"):
            GPModel(train_config=GPTrainConfig(optimizer_type="bogus"), device="cpu")

    def test_invalid_max_attempts_raises(self):
        """max_attempts < 1 raises ValueError at construction."""
        with pytest.raises(ValueError, match="max_attempts"):
            GPModel(train_config=GPTrainConfig(max_attempts=0), device="cpu")

    def test_invalid_dtype_raises(self):
        """An unknown dtype string raises ValueError at construction."""
        with pytest.raises(ValueError, match="not a valid floating-point torch dtype"):
            GPModel(train_config=GPTrainConfig(dtype="not_a_dtype"), device="cpu")

    def test_non_floating_dtype_raises(self):
        """A valid but non-floating-point dtype raises ValueError at construction."""
        with pytest.raises(ValueError, match="not a valid floating-point torch dtype"):
            GPModel(train_config=GPTrainConfig(dtype="int64"), device="cpu")

    def test_label_dtype_mismatch_raises(self):
        """A label_dtype differing from dtype raises ValueError at construction."""
        with pytest.raises(ValueError, match="label_dtype .* must match GPTrainConfig.dtype"):
            GPModel(
                train_config=GPTrainConfig(label_dtype=torch.float32, dtype="float64"),
                device="cpu",
            )

    def test_label_dtype_matching_dtype_accepted(self):
        """A label_dtype equal to the resolved dtype is accepted."""
        model = GPModel(
            train_config=GPTrainConfig(label_dtype=torch.float64, dtype="float64"),
            device="cpu",
        )
        assert model._dtype == torch.float64

    def test_nan_loss_raises_and_resets_state(
        self, precomputed_gp_model, tabular_data, monkeypatch
    ):
        """A non-finite final loss raises RuntimeError and resets model state."""

        def fake_optimize(self, train_x, train_y):
            return {
                "final_mll": float("nan"),
                "final_loss": float("nan"),
                "num_iterations": 1,
            }

        monkeypatch.setattr(GPModel, "_optimize_hyperparameters", fake_optimize)
        with pytest.raises(RuntimeError, match="NaN/Inf loss"):
            precomputed_gp_model.train(tabular_data, None)

        assert precomputed_gp_model.gp_model is None
        assert precomputed_gp_model.likelihood is None
        assert precomputed_gp_model.train_x is None
        assert precomputed_gp_model.train_y is None
        assert precomputed_gp_model.feature_dim is None
        assert precomputed_gp_model._input_transform is None
        assert precomputed_gp_model._output_standardiser is None
        assert precomputed_gp_model.training_metrics == {}
        assert precomputed_gp_model._epoch_metrics == []

    def test_botorch_model_property(self, precomputed_gp_model, tabular_data):
        """botorch_model raises before train and returns SingleTaskGP after."""
        with pytest.raises(RuntimeError, match="must be trained"):
            _ = precomputed_gp_model.botorch_model

        precomputed_gp_model.train(tabular_data, None)
        assert isinstance(precomputed_gp_model.botorch_model, SingleTaskGP)

    def test_default_noise_prior_is_registered(self, precomputed_gp_model, tabular_data):
        """The default noise_prior registers a prior on the likelihood."""
        precomputed_gp_model.train(tabular_data, None)
        prior_names = {name for name, *_ in precomputed_gp_model.likelihood.named_priors()}
        assert any("noise_prior" in name for name in prior_names)

    def test_noise_prior_none_gives_prior_free_likelihood(self, tabular_data):
        """noise_prior=None produces a likelihood with no registered priors."""
        model = GPModel(
            model_config=GPModelConfig(kernel_type="rbf", ard=False, noise_prior=None),
            train_config=GPTrainConfig(num_iterations=5),
            featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
            device="cpu",
        )
        model.train(tabular_data, None)
        prior_names = {name for name, *_ in model.likelihood.named_priors()}
        assert not any("noise_prior" in name for name in prior_names)

    def test_predict_variance_includes_noise(self, precomputed_gp_model, tabular_data):
        """predict() variances exceed the noise-free latent posterior variances."""
        precomputed_gp_model.train(tabular_data, None)
        predictions = precomputed_gp_model.predict(tabular_data.candidates)

        # Build the same normalised test tensor predict() uses internally
        test_x_np = precomputed_gp_model.featurise(tabular_data.candidates).cpu().numpy()
        test_x_np = precomputed_gp_model._input_transform.transform(test_x_np)
        test_x = torch.tensor(test_x_np, dtype=torch.float64)

        with torch.no_grad():
            latent = precomputed_gp_model.botorch_model.posterior(test_x)
            latent_means = latent.mean.squeeze(-1).cpu().numpy()
            latent_vars = latent.variance.squeeze(-1).cpu().numpy()

        # Apply the same inverse transform predict() applies
        _, latent_vars = precomputed_gp_model._output_standardiser.inverse_transform(
            latent_means, latent_vars
        )
        assert np.all(predictions.variances > latent_vars)

    def test_predict_dimension_mismatch_raises(self, precomputed_gp_model, tabular_data):
        """Predicting on candidates with the wrong feature dimension raises ValueError."""
        precomputed_gp_model.train(tabular_data, None)
        bad_candidates = [
            Candidate(data=np.random.rand(4), modality=Modality.TABULAR) for _ in range(3)
        ]
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            precomputed_gp_model.predict(bad_candidates)

    def test_tabular_precomputed_end_to_end(self, precomputed_gp_model, tabular_data):
        """Tabular numpy candidates work end-to-end with featurizer_type='precomputed'."""
        precomputed_gp_model.train(tabular_data, None)
        predictions = precomputed_gp_model.predict(tabular_data.candidates)
        assert predictions.means.shape == (len(tabular_data),)
        assert predictions.variances.shape == (len(tabular_data),)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(predictions.variances >= 0)

    def test_default_dtype_is_float64(self, precomputed_gp_model, tabular_data):
        """The trained botorch_model parameters default to float64."""
        precomputed_gp_model.train(tabular_data, None)
        for param in precomputed_gp_model.botorch_model.parameters():
            assert param.dtype == torch.float64

    @pytest.mark.filterwarnings("ignore::botorch.exceptions.InputDataWarning")
    def test_float32_end_to_end(self, tabular_data):
        """dtype='float32' trains, predicts, and keeps parameters in float32.

        BoTorch's float64 recommendation (InputDataWarning) is expected here
        since float32 is the explicit point of this test.
        """
        model = GPModel(
            model_config=GPModelConfig(kernel_type="rbf", ard=False),
            train_config=GPTrainConfig(num_iterations=10, dtype="float32"),
            featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
            device="cpu",
        )
        model.train(tabular_data, None)
        predictions = model.predict(tabular_data.candidates)
        assert predictions.means.shape == (len(tabular_data),)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(predictions.variances >= 0)
        for param in model.botorch_model.parameters():
            assert param.dtype == torch.float32


class TestGPModelViaSurrogate:
    """Test GPModel accessed through the Surrogate wrapper."""

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in multiply"
        ":RuntimeWarning:alf_core.utils.metrics.regression"
    )
    def test_surrogate_predict_returns_finite_results(self, trained_surrogate, branin_dataset):
        """Predictions from a trained Surrogate have finite means and non-negative variances.

        The rank-space ECE metric computed during fitting hits `inf * sqrt(0)`
        (NaN) at its final confidence-grid point when Monte-Carlo rank
        variances are exactly zero, emitting an expected RuntimeWarning.
        """
        predictions = trained_surrogate.predict(branin_dataset.test_dataset.candidates)

        assert predictions.means is not None
        assert predictions.variances is not None
        assert predictions.means.shape == (len(branin_dataset.test_dataset.candidates),)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(predictions.variances >= 0)
