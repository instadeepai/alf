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

"""Public API for the metrics package.

Re-exports all metric functions, decorators, and registry instances from the
sub-modules so that existing imports of `alf_core.utils.metrics` continue to
work without change.
"""

from alf_core.utils.metrics.acquisition_batch import (
    compute_recall,
    compute_regret,
    intra_batch_diversity,
)
from alf_core.utils.metrics.aggregate import (
    auc_top_k,
    compute_aggregate_metrics,
    compute_experiment_summary,
)
from alf_core.utils.metrics.base import (
    ClassificationMetricRegistry,
    RegressionMetricRegistry,
    check_inputs,
    check_variance_validity,
    classification_metric_registry,
    regression_metric_registry,
    require_min_samples,
)
from alf_core.utils.metrics.calibration import (
    expected_calibration_error,
    rank_expected_calibration_error,
)
from alf_core.utils.metrics.classification import (
    accuracy,
    auc_roc,
    f1,
    precision,
    recall,
    register_classification_metric,
)
from alf_core.utils.metrics.regression import (
    coverage,
    hit_rate,
    monte_carlo_ranking,
    mse,
    nll_gaussian,
    pairwise_xent,
    pearson,
    rank_coverage,
    rank_width,
    register_no_variance_required,
    register_requires_variance,
    regret_ucb_alpha,
    regret_ucb_alpha_sweep,
    residual_pearson,
    residual_spearman,
    spearman,
    top_k_max,
    top_k_mean,
    width,
)
