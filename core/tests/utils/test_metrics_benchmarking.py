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

import alf_core.utils.metrics.benchmarking as bm_module
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
        fresh_registry = BenchmarkingMetricRegistry()

        def my_isolated_test_metric(means, targets):
            """Dummy metric. Returns: dict."""
            return {"my_isolated_test_metric": 0.0}

        fresh_registry.register(my_isolated_test_metric.__name__, my_isolated_test_metric)
        assert "my_isolated_test_metric" in fresh_registry.get_metrics()

    def test_decorated_function_is_callable(self, monkeypatch):
        """Decorated and registered function remains callable."""
        isolated_registry = BenchmarkingMetricRegistry()
        monkeypatch.setattr(bm_module, "benchmarking_metric_registry", isolated_registry)

        @register_benchmarking_metric
        def another_isolated_metric(means, targets):
            """Dummy metric. Returns: dict."""
            return {"another_isolated_metric": float(len(means))}

        result = another_isolated_metric(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert result == {"another_isolated_metric": 3.0}
        assert "another_isolated_metric" in isolated_registry.get_metrics()


class TestGlobalBenchmarkingRegistryInstance:
    """Tests that the global instance is accessible from the module."""

    def test_global_instance_is_benchmarking_metric_registry(self):
        """benchmarking_metric_registry is a BenchmarkingMetricRegistry instance."""
        assert isinstance(benchmarking_metric_registry, BenchmarkingMetricRegistry)
