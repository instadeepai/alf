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

"""Tests for the component registry and its config-coercion builder."""

import dataclasses

import pytest
from alf_benchmark.registry import Registry, default_registry, register
from pydantic import BaseModel


class _FakeDatasetConfig(BaseModel):
    """Single pydantic config (dataset-shaped component)."""

    rows: int
    label: str = "x"


class _FakeDataset:
    """Component built from a single config object."""

    def __init__(self, config: _FakeDatasetConfig) -> None:
        """Store the config.

        Args:
            config: The dataset config.
        """
        self.config = config


@dataclasses.dataclass
class _FakeModelConfig:
    """Model architecture config (dataclass)."""

    width: int = 8


@dataclasses.dataclass
class _FakeTrainConfig:
    """Model training config (dataclass)."""

    epochs: int = 1


class _FakeModel:
    """Component with two config objects plus a scalar kwarg."""

    def __init__(
        self,
        name: str = "fake",
        model_config: _FakeModelConfig | None = None,
        train_config: _FakeTrainConfig | None = None,
    ) -> None:
        """Store the configs and name.

        Args:
            name: Model name (scalar kwarg).
            model_config: Architecture config.
            train_config: Training config.
        """
        self.name = name
        self.model_config = model_config or _FakeModelConfig()
        self.train_config = train_config or _FakeTrainConfig()


class _FakeAcq:
    """Component configured with a scalar kwarg only."""

    def __init__(self, alpha: float) -> None:
        """Store the scalar.

        Args:
            alpha: Exploration weight.
        """
        self.alpha = alpha


def test_build_single_config_object_from_flat_dict():
    """A single-config component receives the flat dict as its config fields."""
    reg = Registry()
    reg.register("alf.datasets", "fake", _FakeDataset)
    obj = reg.build("alf.datasets", "fake", {"rows": 5, "label": "y"})
    assert isinstance(obj, _FakeDataset)
    assert obj.config.rows == 5
    assert obj.config.label == "y"


def test_build_multi_config_and_scalar_kwargs():
    """Config-object kwargs are coerced from sub-dicts; scalars pass through."""
    reg = Registry()
    reg.register("alf.models", "fake", _FakeModel)
    obj = reg.build(
        "alf.models",
        "fake",
        {"name": "m", "model_config": {"width": 32}, "train_config": {"epochs": 3}},
    )
    assert obj.name == "m"
    assert obj.model_config.width == 32
    assert obj.train_config.epochs == 3


def test_build_scalar_only_component():
    """A scalar-only component receives plain keyword arguments."""
    reg = Registry()
    reg.register("alf.acquisition_functions", "fake_acq", _FakeAcq)
    obj = reg.build("alf.acquisition_functions", "fake_acq", {"alpha": 2.5})
    assert obj.alpha == 2.5


def test_build_with_defaults_when_config_empty():
    """An empty config uses the component's own defaults."""
    reg = Registry()
    reg.register("alf.models", "fake", _FakeModel)
    obj = reg.build("alf.models", "fake")
    assert obj.name == "fake"
    assert obj.model_config.width == 8


def test_build_config_object_kwarg_requires_mapping():
    """A config-object kwarg given a non-mapping value raises a clear TypeError."""
    reg = Registry()
    reg.register("alf.models", "fake", _FakeModel)
    with pytest.raises(TypeError, match="model_config"):
        reg.build("alf.models", "fake", {"model_config": 5})


def test_register_decorator_registers_into_registry():
    """The register decorator adds a class to the given registry."""
    reg = Registry()

    @register("alf.models", "decorated", registry=reg)
    class _Decorated:
        """A trivially registered component."""

        def __init__(self) -> None:
            """No-op constructor."""

    assert reg.get("alf.models", "decorated") is _Decorated


def test_missing_name_raises_keyerror_listing_available():
    """Requesting an unknown name raises KeyError listing what is available."""
    reg = Registry()
    reg.register("alf.models", "fake", _FakeModel)
    with pytest.raises(KeyError) as excinfo:
        reg.get("alf.models", "missing")
    assert "fake" in str(excinfo.value)


def test_entry_points_discovered_for_real_components():
    """The real alf-tools components are discoverable via entry points."""
    reg = default_registry()
    assert "cnn" in reg.names("alf.models")
    assert "gfp" in reg.names("alf.datasets")
    assert "ucb" in reg.names("alf.acquisition_functions")
    assert "dataset_search" in reg.names("alf.searches")


class _ExternalModel:
    """Stands in for a model defined in a third-party package."""

    def __init__(self) -> None:
        """No-op constructor."""


class _FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(self, name: str, component_cls: type) -> None:
        """Store the name and the class its ``load`` returns.

        Args:
            name: Entry-point name.
            component_cls: Class returned by :meth:`load`.
        """
        self.name = name
        self._component_cls = component_cls

    def load(self) -> type:
        """Return the component class (as a real entry point would).

        Returns:
            The component class.
        """
        return self._component_cls


def test_external_entry_point_discovered_and_built(monkeypatch):
    """A component exposed by an external package's entry point is discovered and built.

    The fake provider returns its class via ``EntryPoint.load`` without importing
    ``alf_benchmark`` -- the adoption-lever property the registry promises.
    """

    def fake_entry_points(group):
        if group == "alf.models":
            return [_FakeEntryPoint("ext_model", _ExternalModel)]
        return []

    monkeypatch.setattr("alf_benchmark.registry.metadata.entry_points", fake_entry_points)
    reg = Registry()
    reg.discover()
    assert "ext_model" in reg.names("alf.models")
    assert isinstance(reg.build("alf.models", "ext_model"), _ExternalModel)
