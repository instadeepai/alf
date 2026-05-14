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
import pytest
from alf_core import Candidate, LabelledCandidates, Predictions, Surrogate
from alf_core.model.base_model import BaseModel
from alf_core.normalizer.normalizer import (
    IdentityNormalizer,
    MinMaxNormalizer,
    Normalizer,
    ZScoreNormalizer,
)


@pytest.fixture
def labels():
    """Return a 1-D array of five float labels.

    Returns:
        Array [1.0, 2.0, 3.0, 4.0, 5.0].
    """
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def constant_labels():
    """Return a 1-D array of three identical float labels.

    Returns:
        Array [3.0, 3.0, 3.0].
    """
    return np.array([3.0, 3.0, 3.0])


class TestIdentityNormalizer:
    """Tests for IdentityNormalizer."""

    def test_transform_is_identity(self, labels):
        """Transform must return the input labels unchanged."""
        n = IdentityNormalizer()
        n.fit(labels)
        np.testing.assert_array_equal(n.transform(labels), labels)

    def test_inverse_transform_is_identity(self, labels):
        """Inverse transform must return the input values unchanged."""
        n = IdentityNormalizer()
        n.fit(labels)
        np.testing.assert_array_equal(n.inverse_transform(labels), labels)

    def test_is_normalizer(self):
        """IdentityNormalizer must be an instance of Normalizer."""
        assert isinstance(IdentityNormalizer(), Normalizer)

    def test_inverse_variance_scale_is_one(self, labels):
        """Inverse variance transform must return variances unchanged."""
        n = IdentityNormalizer()
        n.fit(labels)
        variances = np.array([1.0, 2.0])
        np.testing.assert_array_equal(n.inverse_transform_variance(variances), variances)


class TestZScoreNormalizer:
    """Tests for ZScoreNormalizer."""

    def test_transform_zero_mean_unit_std(self, labels):
        """Transformed labels must have zero mean and unit standard deviation."""
        n = ZScoreNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        np.testing.assert_almost_equal(transformed.mean(), 0.0, decimal=10)
        np.testing.assert_almost_equal(transformed.std(), 1.0, decimal=10)

    def test_inverse_transform_recovers_original(self, labels):
        """Inverse transform must recover the original labels after standardisation."""
        n = ZScoreNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        recovered = n.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(recovered, labels)

    def test_inverse_variance_scales_by_std_squared(self, labels):
        """Inverse variance must scale by the squared standard deviation."""
        n = ZScoreNormalizer()
        n.fit(labels)
        variances = np.array([1.0])
        expected = variances * (labels.std() ** 2)
        np.testing.assert_array_almost_equal(n.inverse_transform_variance(variances), expected)

    def test_constant_labels_does_not_divide_by_zero(self, constant_labels):
        """Transform of constant labels must return zeros without dividing by zero."""
        n = ZScoreNormalizer()
        n.fit(constant_labels)
        transformed = n.transform(constant_labels)
        np.testing.assert_array_equal(transformed, np.zeros(3))

    def test_is_normalizer(self):
        """ZScoreNormalizer must be an instance of Normalizer."""
        assert isinstance(ZScoreNormalizer(), Normalizer)

    def test_fit_required_before_transform(self):
        """Transform must raise RuntimeError if fit has not been called."""
        n = ZScoreNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            n.transform(np.array([1.0]))

    def test_constant_labels_inverse_transform_returns_constant(self, constant_labels):
        """Inverse transform of zeros must return the constant label value."""
        n = ZScoreNormalizer()
        n.fit(constant_labels)
        recovered = n.inverse_transform(np.zeros(3))
        np.testing.assert_array_almost_equal(recovered, constant_labels)

    def test_constant_labels_inverse_transform_variance_returns_zeros(self, constant_labels):
        """Inverse variance of constant labels must return zeros."""
        n = ZScoreNormalizer()
        n.fit(constant_labels)
        result = n.inverse_transform_variance(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.zeros(2))


class TestMinMaxNormalizer:
    """Tests for MinMaxNormalizer."""

    def test_transform_to_zero_one_range(self, labels):
        """Transformed labels must span the [0, 1] range."""
        n = MinMaxNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        np.testing.assert_almost_equal(transformed.min(), 0.0)
        np.testing.assert_almost_equal(transformed.max(), 1.0)

    def test_inverse_transform_recovers_original(self, labels):
        """Inverse transform must recover the original labels after min-max scaling."""
        n = MinMaxNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        recovered = n.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(recovered, labels)

    def test_constant_labels_does_not_divide_by_zero(self, constant_labels):
        """Transform of constant labels must return zeros without dividing by zero."""
        n = MinMaxNormalizer()
        n.fit(constant_labels)
        transformed = n.transform(constant_labels)
        np.testing.assert_array_equal(transformed, np.zeros(3))

    def test_is_normalizer(self):
        """MinMaxNormalizer must be an instance of Normalizer."""
        assert isinstance(MinMaxNormalizer(), Normalizer)

    def test_fit_required_before_transform(self):
        """Transform must raise RuntimeError if fit has not been called."""
        n = MinMaxNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            n.transform(np.array([1.0]))

    def test_transform_interior_values(self, labels):
        """Transformed values must follow the linear [0, 1] mapping."""
        n = MinMaxNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        np.testing.assert_array_almost_equal(transformed, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_constant_labels_inverse_transform_returns_constant(self, constant_labels):
        """Inverse transform of zeros must return the constant label value."""
        n = MinMaxNormalizer()
        n.fit(constant_labels)
        recovered = n.inverse_transform(np.zeros(3))
        np.testing.assert_array_almost_equal(recovered, constant_labels)

    def test_inverse_variance_scales_by_range_squared(self, labels):
        """Inverse variance must scale by the squared label range."""
        n = MinMaxNormalizer()
        n.fit(labels)
        variances = np.array([1.0])
        scale = labels.max() - labels.min()
        expected = variances * (scale**2)
        np.testing.assert_array_almost_equal(n.inverse_transform_variance(variances), expected)

    def test_constant_labels_inverse_transform_variance_returns_zeros(self, constant_labels):
        """Inverse variance of constant labels must return zeros."""
        n = MinMaxNormalizer()
        n.fit(constant_labels)
        result = n.inverse_transform_variance(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.zeros(2))


class _ConstantModel(BaseModel):
    """Test double: always predicts 0 with variance 1."""

    def featurise(self, inputs):
        """No-op featurise.

        Args:
            inputs: Raw inputs.

        Returns:
            Inputs unchanged.
        """
        return inputs

    def train(self, train_data, val_data=None):
        """Store train labels for inspection.

        Args:
            train_data: Labelled candidates for training.
            val_data: Ignored.
        """
        self._train_labels = train_data.labels.copy()

    def predict(self, candidate_points):
        """Always predict 0 mean with variance 1.

        Args:
            candidate_points: List of candidates to predict for.

        Returns:
            Predictions with zero means and unit variances.
        """
        return Predictions(
            means=np.zeros(len(candidate_points)),
            variances=np.ones(len(candidate_points)),
        )

    def sample(self, *args, **kwargs):
        """Not implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


@pytest.fixture
def labelled_data():
    """Return five candidates with labels [1, 2, 3, 4, 5].

    Returns:
        LabelledCandidates with five sequence candidates and labels [1..5].
    """
    candidates = [Candidate(data="A", modality="sequence") for _ in range(5)]
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return LabelledCandidates(candidates, labels)


class TestSurrogateNormalization:
    """Tests for Surrogate normalizer integration."""

    def test_default_normalizer_is_identity(self, labelled_data):
        """Surrogate without explicit normalizer must use IdentityNormalizer."""
        surrogate = Surrogate(model=_ConstantModel())
        assert isinstance(surrogate.normalizer, IdentityNormalizer)

    def test_zscore_normalizer_inverse_transforms_predictions(self, labelled_data):
        """Means predicted in normalised space must be inverse-transformed to label space."""
        surrogate = Surrogate(model=_ConstantModel(), normalizer=ZScoreNormalizer())
        surrogate.fit(labelled_data, labelled_data)
        preds = surrogate.predict(labelled_data.candidates)
        # Model predicts 0 in normalised space; inverse of 0 with z-score = mean of labels
        expected_mean = float(np.mean(labelled_data.labels))
        np.testing.assert_almost_equal(preds.means[0], expected_mean)

    def test_zscore_normalizer_inverse_transforms_variances(self, labelled_data):
        """Variances predicted in normalised space must be inverse-transformed to label space."""
        surrogate = Surrogate(model=_ConstantModel(), normalizer=ZScoreNormalizer())
        surrogate.fit(labelled_data, labelled_data)
        preds = surrogate.predict(labelled_data.candidates)
        # Model predicts variance=1 in normalised space; inverse = std^2
        expected_var = float(np.std(labelled_data.labels) ** 2)
        np.testing.assert_almost_equal(preds.variances[0], expected_var)

    def test_original_labels_not_mutated(self, labelled_data):
        """fit() must not mutate the labels of the input LabelledCandidates."""
        original_labels = labelled_data.labels.copy()
        surrogate = Surrogate(model=_ConstantModel(), normalizer=ZScoreNormalizer())
        surrogate.fit(labelled_data, labelled_data)
        np.testing.assert_array_equal(labelled_data.labels, original_labels)
