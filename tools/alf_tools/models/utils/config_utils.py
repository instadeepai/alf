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

"""Config instantiation utilities for configurations with _target_ keys."""

import importlib

_ALLOWED_MODULES = frozenset({"gpytorch.priors", "gpytorch.constraints"})


def build_from_target(cfg: dict | object | None) -> object | None:
    """Instantiate a GPyTorch object from a _target_ config dict.

    Supports any GPyTorch prior or constraint. The dict must contain a
    `_target_` key with a fully-qualified class path; all other keys
    are passed as constructor kwargs.

    Already-instantiated objects (e.g. a ``gpytorch.priors.Prior`` instance)
    are passed through unchanged, which preserves backward compatibility with
    code that constructs priors directly rather than via the ``_target_`` dict
    format.

    Args:
        cfg: One of:
            - A dict with `_target_` (e.g. `"gpytorch.priors.LogNormalPrior"`)
              and any constructor kwargs.
            - An already-instantiated GPyTorch prior or constraint object
              (passed through unchanged).
            - `None`, which returns `None`.

    Raises:
        ValueError: If `_target_` is not in the allowed module list.

    Returns:
        Instantiated object, or `None` if `cfg` is `None`.

    Example::

        prior = build_from_target({
            "_target_": "gpytorch.priors.GammaPrior",
            "concentration": 3.0,
            "rate": 6.0,
        })
    """
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        # Already instantiated — pass through for backward compatibility.
        return cfg
    cfg = dict(cfg)  # don't mutate the caller's dict
    target = cfg.pop("_target_")

    module_path, cls_name = target.rsplit(".", 1)
    if module_path not in _ALLOWED_MODULES and not any(
        module_path.startswith(prefix + ".") for prefix in _ALLOWED_MODULES
    ):
        raise ValueError(
            f"build_from_target: _target_ '{target}' is not in the allowed module list. "
            f"Only gpytorch.priors.* and gpytorch.constraints.* are permitted."
        )
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**cfg)
