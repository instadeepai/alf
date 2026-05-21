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

"""Tests for the benchmarking metric registry scaffold."""

import numpy as np
from alf_core.utils.metrics.benchmarking import (
    BenchmarkingMetricRegistry,
    benchmarking_metric_registry,
    register_benchmarking_metric,
)


class TestBenchmarkingMetricRegistry:
    """Tests for BenchmarkingMetricRegistry."""

    def test_register_and_retrieve(self):
        """A registered function is returned by get_metrics."""
        reg = BenchmarkingMetricRegistry()
        reg.register("my_metric", lambda m, t: {"my_metric": 0.0})
        assert "my_metric" in reg.get_metrics()

    def test_empty_registry(self):
        """Fresh registry has empty metrics dict."""
        reg = BenchmarkingMetricRegistry()
        assert reg.get_metrics() == {}


class TestRegisterBenchmarkingMetric:
    """Tests for register_benchmarking_metric decorator."""

    def test_decorator_registers_function(self):
        """Decorated function is added to the global benchmarking registry."""
        initial_count = len(benchmarking_metric_registry.get_metrics())

        @register_benchmarking_metric
        def my_test_metric(means, targets):
            """Dummy benchmarking metric.

            Returns:
                dict with a single key "my_test_metric".
            """
            return {"my_test_metric": 0.0}

        assert len(benchmarking_metric_registry.get_metrics()) == initial_count + 1
        assert "my_test_metric" in benchmarking_metric_registry.get_metrics()

    def test_decorated_function_is_callable(self):
        """Decorated and registered function remains callable."""

        @register_benchmarking_metric
        def another_metric(means, targets):
            """Dummy benchmarking metric.

            Returns:
                dict with a single key "another_metric".
            """
            return {"another_metric": float(len(means))}

        result = another_metric(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert result == {"another_metric": 3.0}


class TestGlobalBenchmarkingRegistryInstance:
    """Tests that the global instance is accessible from the module."""

    def test_global_instance_is_benchmarking_metric_registry(self):
        """benchmarking_metric_registry is a BenchmarkingMetricRegistry instance."""
        assert isinstance(benchmarking_metric_registry, BenchmarkingMetricRegistry)
