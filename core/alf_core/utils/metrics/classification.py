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

"""Classification metrics.

All metrics are automatically registered in ``classification_metric_registry``
at import time via the ``@register_classification_metric`` decorator.
"""

import logging
from functools import wraps
from typing import Callable

import numpy as np
from alf_core.utils.metrics.base import classification_metric_registry, require_min_samples
from jaxtyping import Float, Int
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("alf-core")


def register_classification_metric(metric_fn: Callable) -> Callable:
    """Decorator to register a classification metric.

    Automatically registers the metric in the classification registry and applies
    basic input validation. The decorated function receives ``(probs, targets)``
    where ``probs`` has shape ``(n_samples, num_classes)`` and ``targets`` has
    shape ``(n_samples,)``.

    Args:
        metric_fn: The metric function to decorate.

    Returns:
        Wrapped metric function with validation and registration.
    """

    @wraps(metric_fn)
    def wrapper(
        probs: Float[np.ndarray, "n_samples num_classes"], targets: Float[np.ndarray, " n_samples"]
    ) -> dict[str, float]:
        assert probs.ndim == 2, (
            f"probs must be with shape (n_samples, num_classes), got shape {probs.shape}"
        )
        assert len(probs) != 0, "Empty input arrays"
        assert probs.shape[0] == targets.shape[0], (
            f"probs and targets batch size mismatch: {probs.shape[0]} vs {targets.shape[0]}"
        )
        targets = targets.astype(int)
        return metric_fn(probs, targets)

    classification_metric_registry.register(metric_fn.__name__, wrapper)
    return wrapper


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


@register_classification_metric
def accuracy(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute classification accuracy.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"accuracy": accuracy float}
    """
    preds = np.argmax(probs, axis=1)
    return {"accuracy": float(accuracy_score(targets, preds))}


@register_classification_metric
def f1(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute macro-averaged F1 score.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"f1": macro F1 float}
    """
    preds = np.argmax(probs, axis=1)
    return {"f1": float(f1_score(targets, preds, average="macro", zero_division=0))}


@register_classification_metric
def precision(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute macro-averaged precision.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"precision": macro precision float}
    """
    preds = np.argmax(probs, axis=1)
    return {"precision": float(precision_score(targets, preds, average="macro", zero_division=0))}


@register_classification_metric
def recall(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute macro-averaged recall.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"recall": macro recall float}
    """
    preds = np.argmax(probs, axis=1)
    return {"recall": float(recall_score(targets, preds, average="macro", zero_division=0))}


@register_classification_metric
@require_min_samples(2)
def auc_roc(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute Area Under the ROC Curve (AUC-ROC).

    For binary classification, uses the positive-class probabilities.
    For multiclass, uses one-vs-rest averaging.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"auc_roc": AUC-ROC float}
    """
    if len(np.unique(targets)) < 2:
        logger.warning("auc_roc: fewer than 2 unique classes in targets — returning empty dict.")
        return {}
    if probs.shape[1] == 2:
        try:
            return {"auc_roc": float(roc_auc_score(targets, probs[:, 1]))}
        except ValueError as e:
            logger.warning(
                "auc_roc: sklearn raised ValueError (likely targets missing a class): %s", e
            )
            return {}
    try:
        return {"auc_roc": float(roc_auc_score(targets, probs, multi_class="ovr"))}
    except ValueError as e:
        logger.warning("auc_roc: sklearn raised ValueError (likely targets missing a class): %s", e)
        return {}
