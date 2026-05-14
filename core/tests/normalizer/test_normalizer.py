import numpy as np
import pytest
from alf_core.normalizer.normalizer import (
    IdentityNormalizer,
    MinMaxNormalizer,
    Normalizer,
    ZScoreNormalizer,
)


@pytest.fixture
def labels():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def constant_labels():
    return np.array([3.0, 3.0, 3.0])


class TestIdentityNormalizer:
    """Tests for IdentityNormalizer."""

    def test_transform_is_identity(self, labels):
        n = IdentityNormalizer()
        n.fit(labels)
        np.testing.assert_array_equal(n.transform(labels), labels)

    def test_inverse_transform_is_identity(self, labels):
        n = IdentityNormalizer()
        n.fit(labels)
        np.testing.assert_array_equal(n.inverse_transform(labels), labels)

    def test_is_normalizer(self):
        assert isinstance(IdentityNormalizer(), Normalizer)

    def test_inverse_variance_scale_is_one(self, labels):
        n = IdentityNormalizer()
        n.fit(labels)
        variances = np.array([1.0, 2.0])
        np.testing.assert_array_equal(n.inverse_transform_variance(variances), variances)


class TestZScoreNormalizer:
    """Tests for ZScoreNormalizer."""

    def test_transform_zero_mean_unit_std(self, labels):
        n = ZScoreNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        np.testing.assert_almost_equal(transformed.mean(), 0.0, decimal=10)
        np.testing.assert_almost_equal(transformed.std(), 1.0, decimal=10)

    def test_inverse_transform_recovers_original(self, labels):
        n = ZScoreNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        recovered = n.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(recovered, labels)

    def test_inverse_variance_scales_by_std_squared(self, labels):
        n = ZScoreNormalizer()
        n.fit(labels)
        variances = np.array([1.0])
        expected = variances * (labels.std() ** 2)
        np.testing.assert_array_almost_equal(
            n.inverse_transform_variance(variances), expected
        )

    def test_constant_labels_does_not_divide_by_zero(self, constant_labels):
        n = ZScoreNormalizer()
        n.fit(constant_labels)
        transformed = n.transform(constant_labels)
        np.testing.assert_array_equal(transformed, np.zeros(3))

    def test_is_normalizer(self):
        assert isinstance(ZScoreNormalizer(), Normalizer)

    def test_fit_required_before_transform(self):
        n = ZScoreNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            n.transform(np.array([1.0]))

    def test_constant_labels_inverse_transform_returns_constant(self, constant_labels):
        n = ZScoreNormalizer()
        n.fit(constant_labels)
        recovered = n.inverse_transform(np.zeros(3))
        np.testing.assert_array_almost_equal(recovered, constant_labels)

    def test_constant_labels_inverse_transform_variance_returns_zeros(self, constant_labels):
        n = ZScoreNormalizer()
        n.fit(constant_labels)
        result = n.inverse_transform_variance(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.zeros(2))


class TestMinMaxNormalizer:
    """Tests for MinMaxNormalizer."""

    def test_transform_to_zero_one_range(self, labels):
        n = MinMaxNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        np.testing.assert_almost_equal(transformed.min(), 0.0)
        np.testing.assert_almost_equal(transformed.max(), 1.0)

    def test_inverse_transform_recovers_original(self, labels):
        n = MinMaxNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        recovered = n.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(recovered, labels)

    def test_constant_labels_does_not_divide_by_zero(self, constant_labels):
        n = MinMaxNormalizer()
        n.fit(constant_labels)
        transformed = n.transform(constant_labels)
        np.testing.assert_array_equal(transformed, np.zeros(3))

    def test_is_normalizer(self):
        assert isinstance(MinMaxNormalizer(), Normalizer)

    def test_fit_required_before_transform(self):
        n = MinMaxNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            n.transform(np.array([1.0]))

    def test_transform_interior_values(self, labels):
        n = MinMaxNormalizer()
        n.fit(labels)
        transformed = n.transform(labels)
        np.testing.assert_array_almost_equal(transformed, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_constant_labels_inverse_transform_returns_constant(self, constant_labels):
        n = MinMaxNormalizer()
        n.fit(constant_labels)
        recovered = n.inverse_transform(np.zeros(3))
        np.testing.assert_array_almost_equal(recovered, constant_labels)

    def test_inverse_variance_scales_by_range_squared(self, labels):
        n = MinMaxNormalizer()
        n.fit(labels)
        variances = np.array([1.0])
        scale = labels.max() - labels.min()
        expected = variances * (scale ** 2)
        np.testing.assert_array_almost_equal(n.inverse_transform_variance(variances), expected)

    def test_constant_labels_inverse_transform_variance_returns_zeros(self, constant_labels):
        n = MinMaxNormalizer()
        n.fit(constant_labels)
        result = n.inverse_transform_variance(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.zeros(2))
