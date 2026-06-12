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
import inspect

import gpytorch

_ALLOWED_MODULES = frozenset({"gpytorch.priors", "gpytorch.constraints"})


def build_from_target(
    cfg: dict | None,
    expected_base: type | tuple[type, ...] | None = None,
) -> object | None:
    """Instantiate a GPyTorch object from a _target_ config dict.

    Supports any GPyTorch prior or constraint. The dict must contain a
    `_target_` key with a fully-qualified class path; all other keys
    are passed as constructor kwargs.
    
    Args:
        cfg: Dict with `_target_` (e.g. `"gpytorch.priors.LogNormalPrior"`)
            and any constructor kwargs. `None` returns `None`.
        expected_base: Base class (or tuple of base classes) the resolved
            target must subclass. When `None`, the target is validated
            against the union of `gpytorch.priors.Prior` and
            `gpytorch.constraints.Interval`.

    Raises:
        ValueError: If `_target_` is missing, not fully qualified, not in
            the allowed module list, names a private attribute (leading
            underscore), does not exist in the module, does not resolve to
            a class, does not subclass
            `expected_base`, or if the constructor rejects the provided
            kwargs.

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
    cfg = dict(cfg)  # don't mutate the caller's dict
    target = cfg.pop("_target_", None)
    if target is None:
        raise ValueError(
            "build_from_target: config dict must contain a '_target_' key "
            f"(e.g. 'gpytorch.priors.LogNormalPrior'). Got keys: {sorted(cfg)}"
        )
    if "." not in target:
        raise ValueError(
            f"build_from_target: '_target_' must be a fully-qualified class path "
            f"(e.g. 'gpytorch.priors.LogNormalPrior'), got {target!r}"
        )

    module_path, cls_name = target.rsplit(".", 1)
    if module_path not in _ALLOWED_MODULES and not any(
        module_path.startswith(prefix + ".") for prefix in _ALLOWED_MODULES
    ):
        raise ValueError(
            f"build_from_target: _target_ '{target}' is not in the allowed module list. "
            f"Only gpytorch.priors.* and gpytorch.constraints.* are permitted."
        )
    if cls_name.startswith("_"):
        raise ValueError(
            f"build_from_target: _target_ '{target}' names a private attribute "
            f"(leading underscore), which is not permitted."
        )
    cls = getattr(importlib.import_module(module_path), cls_name, None)

    if expected_base is None:
        expected_base = (gpytorch.priors.Prior, gpytorch.constraints.Interval)
    if not inspect.isclass(cls) or not issubclass(cls, expected_base):
        raise ValueError(
            f"build_from_target: _target_ '{target}' does not resolve to a "
            f"subclass of {expected_base!r}."
        )

    try:
        return cls(**cfg)
    except TypeError as e:
        raise ValueError(
            f"build_from_target: failed to instantiate '{target}' with kwargs {sorted(cfg)}: {e}"
        ) from e
