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

"""Name-based component registry backed by Python entry points.

Providers (``alf_tools`` and external packages) expose components under
entry-point groups such as ``alf.models``. The registry discovers them by name
and builds configured instances from plain dictionaries, so providers register
components *without importing ``alf_benchmark``* — the key adoption lever.

A component's config classes are inferred from its ``__init__`` annotations:
parameters whose type is a dataclass or pydantic model (e.g. ``model_config``)
are coerced from sub-dictionaries; remaining parameters are passed as scalar
keyword arguments. This keeps the alf-core/alf-tools classes untouched.
"""

import dataclasses
import inspect
import logging
import typing
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import Any

import pydantic

logger = logging.getLogger("alf-benchmark")

# Entry-point groups that providers register components under. "alf.oracles" is
# reserved: offline oracles bind to the dataset and online oracles are built from
# "alf.models", so nothing registers under it yet.
GROUPS: tuple[str, ...] = (
    "alf.models",
    "alf.datasets",
    "alf.acquisition_functions",
    "alf.searches",
    "alf.oracles",
)


@dataclasses.dataclass(frozen=True)
class RegistryEntry:
    """A registered component.

    Attributes:
        group: Entry-point group the component belongs to (e.g. ``"alf.models"``).
        name: Short name used to reference the component (e.g. ``"cnn"``).
        load: Zero-argument callable returning the component class. For entry
            points this is ``EntryPoint.load`` (imported lazily on first build).
    """

    group: str
    name: str
    load: Callable[[], type]


def _unwrap_optional(annotation: Any) -> list[Any]:
    """Return the non-``None`` members of a ``Union``/``Optional`` annotation.

    Args:
        annotation: A resolved type annotation.

    Returns:
        The candidate types: the union members excluding ``NoneType``, or the
        annotation itself when it is not a union.
    """
    args = typing.get_args(annotation)
    if not args:
        return [annotation]
    return [arg for arg in args if arg is not type(None)]


def _config_class(annotation: Any) -> type | None:
    """Return the config class for an annotation, if it denotes one.

    A config class is a dataclass or a pydantic model (the two config shapes
    used across alf-core/alf-tools).

    Args:
        annotation: A resolved ``__init__`` parameter annotation.

    Returns:
        The config class, or ``None`` if the parameter is a scalar argument.
    """
    for candidate in _unwrap_optional(annotation):
        if isinstance(candidate, type) and dataclasses.is_dataclass(candidate):
            return candidate
        if isinstance(candidate, type) and issubclass(candidate, pydantic.BaseModel):
            return candidate
    return None


def _resolve_config_classes(component_cls: type) -> tuple[dict[str, type], set[str]]:
    """Introspect a component's ``__init__`` to classify its parameters.

    Args:
        component_cls: The component class to introspect.

    Returns:
        A tuple ``(config_classes, scalar_params)`` where ``config_classes`` maps
        each config-object parameter name to its config class, and
        ``scalar_params`` is the set of remaining (scalar) parameter names.
    """
    try:
        hints = typing.get_type_hints(component_cls.__init__)  # type: ignore[misc]
    except Exception:
        # Fall back to raw annotations if forward references cannot be resolved.
        hints = {}
    # Signature of the class omits ``self`` and reflects the constructor.
    signature = inspect.signature(component_cls)
    config_classes: dict[str, type] = {}
    scalar_params: set[str] = set()
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(param.name, param.annotation)
        config_cls = _config_class(annotation)
        if config_cls is not None:
            config_classes[param.name] = config_cls
        else:
            scalar_params.add(param.name)
    return config_classes, scalar_params


def _instantiate_config(config_cls: type, values: Mapping[str, Any]) -> Any:
    """Construct a config object from a mapping of field values.

    Args:
        config_cls: A dataclass or pydantic config class.
        values: Field values to pass as keyword arguments.

    Returns:
        The constructed config instance.
    """
    return config_cls(**dict(values))


class Registry:
    """Resolves and builds alf components by ``(group, name)``."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._entries: dict[tuple[str, str], RegistryEntry] = {}
        self._discovered = False

    def register(self, group: str, name: str, component_cls: type) -> None:
        """Register a component class directly (in-process, for tests or plugins).

        Args:
            group: Entry-point group (e.g. ``"alf.models"``).
            name: Short name to reference the component by.
            component_cls: The component class to register.
        """
        self._entries[(group, name)] = RegistryEntry(group, name, lambda: component_cls)

    def unregister(self, group: str, name: str) -> None:
        """Remove a registered component if present (no error if absent).

        Args:
            group: Entry-point group (e.g. ``"alf.models"``).
            name: Short name of the component to remove.
        """
        self._entries.pop((group, name), None)

    def discover(self) -> None:
        """Discover components from installed entry points.

        Idempotent and cheap: entry points are recorded but not imported until
        :meth:`get` or :meth:`build` is called, so missing optional dependencies
        do not break discovery. Locally registered entries are never overwritten.
        """
        for group in GROUPS:
            for entry_point in metadata.entry_points(group=group):
                key = (group, entry_point.name)
                if key in self._entries:
                    continue
                self._entries[key] = RegistryEntry(group, entry_point.name, entry_point.load)
        self._discovered = True

    def _ensure_discovered(self) -> None:
        """Run entry-point discovery once if it has not happened yet."""
        if not self._discovered:
            self.discover()

    def names(self, group: str) -> list[str]:
        """List the registered component names in a group.

        Args:
            group: Entry-point group to list.

        Returns:
            Sorted component names registered under the group.
        """
        self._ensure_discovered()
        return sorted(name for (grp, name) in self._entries if grp == group)

    def get(self, group: str, name: str) -> type:
        """Load and return a registered component class.

        Args:
            group: Entry-point group (e.g. ``"alf.models"``).
            name: Short name of the component.

        Returns:
            The component class.

        Raises:
            KeyError: If no component is registered under ``(group, name)``.
        """
        self._ensure_discovered()
        key = (group, name)
        if key not in self._entries:
            available = ", ".join(self.names(group)) or "<none>"
            raise KeyError(
                f"No component '{name}' registered in group '{group}'. Available: {available}"
            )
        return self._entries[key].load()

    def build(self, group: str, name: str, config: Mapping[str, Any] | None = None) -> Any:
        """Build a configured component instance from a config dictionary.

        Coercion rules, inferred from the component's ``__init__`` signature:

        - A component with a single config-object parameter and no scalar
          parameters (e.g. a dataset taking one ``config``) receives the whole
          config dict as that config object's fields.
        - Otherwise, dict keys map to ``__init__`` keyword arguments;
          config-object parameters are built from their sub-dictionaries and
          scalar parameters are passed through unchanged.

        Args:
            group: Entry-point group (e.g. ``"alf.models"``).
            name: Short name of the component.
            config: Configuration dictionary (defaults to empty).

        Returns:
            The constructed component instance.

        Raises:
            KeyError: If no component is registered under ``(group, name)``.
            TypeError: If a config-object parameter is given a non-mapping value.
        """
        component_cls = self.get(group, name)
        config = dict(config or {})
        config_classes, scalar_params = _resolve_config_classes(component_cls)

        # Single config-object component (e.g. datasets): the dict is the config's fields.
        if len(config_classes) == 1 and not scalar_params:
            ((param_name, config_cls),) = config_classes.items()
            if param_name not in config:
                return component_cls(**{param_name: _instantiate_config(config_cls, config)})

        kwargs: dict[str, Any] = {}
        for key, value in config.items():
            if key in config_classes:
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"Component '{name}' parameter '{key}' expects a config mapping for "
                        f"{config_classes[key].__name__}, got {type(value).__name__}."
                    )
                kwargs[key] = _instantiate_config(config_classes[key], value)
            else:
                kwargs[key] = value
        return component_cls(**kwargs)


_DEFAULT_REGISTRY = Registry()


def default_registry() -> Registry:
    """Return the process-wide default registry (entry points discovered lazily).

    Returns:
        The shared :class:`Registry` instance.
    """
    _DEFAULT_REGISTRY._ensure_discovered()
    return _DEFAULT_REGISTRY


def register(group: str, name: str, registry: Registry | None = None) -> Callable[[type], type]:
    """Class decorator that registers a component in a registry.

    Args:
        group: Entry-point group (e.g. ``"alf.models"``).
        name: Short name to reference the component by.
        registry: Registry to register into (defaults to the shared registry).

    Returns:
        A decorator that registers the class and returns it unchanged.
    """
    target = registry if registry is not None else _DEFAULT_REGISTRY

    def decorator(component_cls: type) -> type:
        target.register(group, name, component_cls)
        return component_cls

    return decorator
