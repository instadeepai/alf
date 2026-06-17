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

"""Smoke tests for alf_core.utils.metrics public API backward-compat."""

from alf_core.utils.metrics import (
    accuracy,
    auc_roc,
    auc_top_k,
    check_inputs,
    check_variance_validity,
    classification_metric_registry,
    coverage,
    expected_calibration_error,
    f1,
    mse,
    pearson,
    precision,
    recall,
    regression_metric_registry,
    spearman,
)


def test_top_level_imports_are_callable():
    """All re-exported symbols from __init__.py are importable and callable/accessible."""
    for symbol in [
        accuracy,
        auc_roc,
        auc_top_k,
        f1,
        precision,
        recall,
        coverage,
        expected_calibration_error,
        mse,
        pearson,
        spearman,
        check_inputs,
        check_variance_validity,
    ]:
        assert callable(symbol)
    assert hasattr(classification_metric_registry, "get_metrics")
    assert hasattr(regression_metric_registry, "get_metrics")
