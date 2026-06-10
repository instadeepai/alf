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

"""Global seeding for reproducible replications.

A dataset's own seed controls only its split. Training introduces additional
randomness (weight init, dropout, mini-batch order), so the runner seeds the
global numpy/torch RNGs per replication to make confidence intervals
reproducible — a requirement for Datasets & Benchmarks grading.
"""

import logging
import os
import random

import numpy as np

logger = logging.getLogger("alf-benchmark")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, numpy, and torch (incl. CUDA) RNGs for one replication.

    Args:
        seed: The seed to apply to all RNGs.
        deterministic: If True, request deterministic algorithms and disable
            cuDNN autotuning where supported. Slower, but removes residual
            nondeterminism; off by default.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415 - imported lazily so seeding works without torch
    except ImportError:
        logger.debug("torch not available; seeded Python and numpy only.")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch builds may not support this; best-effort only.
            logger.debug("torch.use_deterministic_algorithms unavailable; skipping.")
