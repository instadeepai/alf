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
from alf_core.model.normaliser import (
    InputNormaliser,
    InputStandardiser,
    OutputStandardiser,
    make_input_transform,
)


class TestOutputStandardiser:
    """Tests for OutputStandardiser Z-score normalisation of labels."""

    def test_transform_produces_zero_mean_unit_variance(self):
        """Transformed labels have zero mean and unit variance."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = OutputStandardiser()
        s.fit(Y)
        Y_t = s.transform(Y)
        assert abs(Y_t.mean()) < 1e-6
        assert abs(Y_t.std() - 1.0) < 1e-6

    def test_inverse_transform_round_trips_mean(self):
        """inverse_transform recovers the original labels from standardised predictions."""
        Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        s = OutputStandardiser()
        s.fit(Y)
        Y_t = s.transform(Y)
        mean_orig, _ = s.inverse_transform(Y_t)
        np.testing.assert_allclose(mean_orig, Y, rtol=1e-5)

    def test_inverse_transform_scales_variance_correctly(self):
        """Variance in standardised space is scaled by std² when inverse-transformed."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = OutputStandardiser()
        s.fit(Y)
        # Variance of 1.0 in standardised space should become std² in original space
        var_std = np.array([1.0, 1.0, 1.0])
        _, var_orig = s.inverse_transform(np.zeros(3), var_std)
        expected = var_std * (s._std**2)
        np.testing.assert_allclose(var_orig, expected, rtol=1e-5)

    def test_inverse_transform_none_variance_returns_none(self):
        """Passing var=None returns None as the second element of the output tuple."""
        Y = np.array([1.0, 2.0, 3.0])
        s = OutputStandardiser()
        s.fit(Y)
        _, var_out = s.inverse_transform(np.zeros(3), None)
        assert var_out is None

    def test_constant_labels_does_not_divide_by_zero(self):
        """Constant training labels clamp std to _MIN_STD, keeping transforms finite."""
        Y = np.array([5.0, 5.0, 5.0, 5.0])
        s = OutputStandardiser()
        s.fit(Y)
        assert s._std >= s._MIN_STD
        Y_t = s.transform(Y)
        assert np.all(np.isfinite(Y_t))

    def test_single_sample(self):
        """A single training sample is handled without errors."""
        Y = np.array([42.0])
        s = OutputStandardiser()
        s.fit(Y)
        Y_t = s.transform(Y)
        assert np.isfinite(Y_t[0])
        mean_orig, _ = s.inverse_transform(Y_t)
        np.testing.assert_allclose(mean_orig, Y, rtol=1e-5)

    def test_transform_before_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        s = OutputStandardiser()
        with pytest.raises(RuntimeError, match="fitted"):
            s.transform(np.array([1.0, 2.0]))

    def test_inverse_transform_before_fit_raises(self):
        """Calling inverse_transform before fit raises RuntimeError."""
        s = OutputStandardiser()
        with pytest.raises(RuntimeError, match="fitted"):
            s.inverse_transform(np.array([1.0, 2.0]))

    def test_is_fitted(self):
        """is_fitted is False before fit() and True after."""
        s = OutputStandardiser()
        assert not s.is_fitted
        s.fit(np.array([1.0, 2.0, 3.0]))
        assert s.is_fitted


class TestInputNormaliser:
    """Tests for InputNormaliser min-max scaling of input features."""

    def test_transform_produces_values_in_zero_one(self):
        """Transformed features lie in [0, 1] for each dimension."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        n = InputNormaliser()
        n.fit(X)
        X_t = n.transform(X)
        assert X_t.min().item() >= 0.0 - 1e-6
        assert X_t.max().item() <= 1.0 + 1e-6

    def test_transform_round_trip_min_max(self):
        """Min sample maps to 0 and max sample maps to 1 per feature after transform."""
        X = np.array([[0.0, 5.0], [5.0, 10.0], [10.0, 15.0]])
        n = InputNormaliser()
        n.fit(X)
        X_t = n.transform(X)
        # Min should transform to 0, max to 1 per feature
        assert abs(X_t[:, 0].min().item()) < 1e-6
        assert abs(X_t[:, 0].max().item() - 1.0) < 1e-6

    def test_constant_feature_does_not_divide_by_zero(self):
        """A constant feature column clamps range to _MIN_RANGE and transforms to 0."""
        X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
        n = InputNormaliser()
        n.fit(X)
        X_t = n.transform(X)
        assert np.all(np.isfinite(X_t))
        # Constant feature should transform to 0
        assert np.all(X_t[:, 0] == 0.0)

    def test_single_sample(self):
        """A single training sample is handled without errors."""
        X = np.array([[1.0, 2.0, 3.0]])
        n = InputNormaliser()
        n.fit(X)
        X_t = n.transform(X)
        assert np.all(np.isfinite(X_t))

    def test_transform_before_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        n = InputNormaliser()
        with pytest.raises(RuntimeError, match="fitted"):
            n.transform(np.array([[1.0, 2.0]]))

    def test_is_fitted(self):
        """is_fitted is False before fit() and True after."""
        n = InputNormaliser()
        assert not n.is_fitted
        n.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert n.is_fitted

    def test_fit_on_3d_tensor(self):
        """InputNormaliser must handle CNN's 3D one-hot tensors (batch, alphabet, seq_len)."""
        X = np.random.rand(8, 20, 10)  # (batch, alphabet_size, seq_len)
        n = InputNormaliser()
        n.fit(X)
        X_t = n.transform(X)
        assert np.all(np.isfinite(X_t))
        assert X_t.shape == X.shape

    def test_inverse_transform_round_trips(self):
        """inverse_transform recovers the original features from normalised ones."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        n = InputNormaliser()
        n.fit(X)
        X_recovered = n.inverse_transform(n.transform(X))
        np.testing.assert_allclose(X_recovered, X, rtol=1e-5)

    def test_inverse_transform_before_fit_raises(self):
        """Calling inverse_transform before fit raises RuntimeError."""
        n = InputNormaliser()
        with pytest.raises(RuntimeError, match="fitted"):
            n.inverse_transform(np.array([[0.0, 1.0]]))


class TestInputStandardiser:
    """Tests for InputStandardiser Z-score standardisation of input features."""

    def test_transform_produces_zero_mean_unit_variance(self):
        """Transformed features have zero mean and unit variance per dimension."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        s = InputStandardiser()
        s.fit(X)
        X_t = s.transform(X)
        np.testing.assert_allclose(X_t.mean(axis=0), np.zeros(2), atol=1e-6)
        np.testing.assert_allclose(X_t.std(axis=0), np.ones(2), atol=1e-6)

    def test_inverse_transform_round_trips(self):
        """inverse_transform recovers the original features from standardised ones."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        s = InputStandardiser()
        s.fit(X)
        X_recovered = s.inverse_transform(s.transform(X))
        np.testing.assert_allclose(X_recovered, X, rtol=1e-5)

    def test_constant_feature_does_not_divide_by_zero(self):
        """A zero-variance feature column has its scale set to 1.0, keeping transforms finite."""
        X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
        s = InputStandardiser()
        s.fit(X)
        # Constant column (index 0) has its scale set to 1.0, not divided by a near-zero std.
        np.testing.assert_allclose(s._std[0], 1.0)
        # Non-constant column (index 1) keeps its actual std.
        np.testing.assert_allclose(s._std[1], np.std(X[:, 1]))
        X_t = s.transform(X)
        assert np.all(np.isfinite(X_t))
        # Constant feature is centred to 0 by mean subtraction.
        np.testing.assert_allclose(X_t[:, 0], np.zeros(3), atol=1e-6)

    def test_single_sample(self):
        """A single training sample is handled without errors (zero variance everywhere)."""
        X = np.array([[1.0, 2.0, 3.0]])
        s = InputStandardiser()
        s.fit(X)
        X_t = s.transform(X)
        assert np.all(np.isfinite(X_t))

    def test_transform_before_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        s = InputStandardiser()
        with pytest.raises(RuntimeError, match="fitted"):
            s.transform(np.array([[1.0, 2.0]]))

    def test_inverse_transform_before_fit_raises(self):
        """Calling inverse_transform before fit raises RuntimeError."""
        s = InputStandardiser()
        with pytest.raises(RuntimeError, match="fitted"):
            s.inverse_transform(np.array([[0.0, 1.0]]))

    def test_is_fitted(self):
        """is_fitted is False before fit() and True after."""
        s = InputStandardiser()
        assert not s.is_fitted
        s.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert s.is_fitted

    def test_fit_on_3d_tensor(self):
        """InputStandardiser must handle CNN's 3D one-hot tensors (batch, alphabet, seq_len)."""
        X = np.random.rand(8, 20, 10)  # (batch, alphabet_size, seq_len)
        s = InputStandardiser()
        s.fit(X)
        X_t = s.transform(X)
        assert np.all(np.isfinite(X_t))
        assert X_t.shape == X.shape


class TestMakeInputTransform:
    """Tests for the make_input_transform factory."""

    def test_minmax_returns_input_normaliser(self):
        """The 'minmax' strategy returns an InputNormaliser."""
        assert isinstance(make_input_transform("minmax"), InputNormaliser)

    def test_zscore_returns_input_standardiser(self):
        """The 'zscore' strategy returns an InputStandardiser."""
        assert isinstance(make_input_transform("zscore"), InputStandardiser)

    def test_unknown_strategy_raises_value_error(self):
        """An unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown input normalisation strategy"):
            make_input_transform("invalid")
