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

import abc
import logging
from typing import Any, Dict, Literal, Union

import matplotlib.pyplot as plt
import neptune
import numpy as np
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig, listconfig
from s3fs import S3FileSystem

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class Logger(abc.ABC):
    """Abstract base class for loggers that can write data and artifacts."""

    @abc.abstractmethod
    def write(self, data: Dict[str, Any], label: str = "", timestep: int | None = None) -> None:
        """Write data to the logger destination.

        Args:
            data: Dictionary of data to write (metrics or figures)
            label: Optional label prefix for the data
            timestep: Optional timestep for the data. If None, uses internal counter
        """
        pass

    def write_artifact(self, file_name: str) -> None:
        """Write an artifact to the logger destination.

        Args:
            file_name: Path to the artifact file to upload
        """
        pass

    def close(self) -> None:
        """Close the logger and clean up resources."""
        pass

    def get_checkpoint(self, file_path: str) -> str | None:
        """Download checkpoint from a run (optional implementation).

        Args:
            file_path: Path to the checkpoint file to download

        Returns:
            Local path to the downloaded checkpoint, or None if not implemented
        """
        return None

    def __enter__(self) -> "Logger":
        """Enter context manager.

        Returns:
            The logger instance for use as a context manager
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close logger.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()


class NeptuneLogger(Logger):
    """Logger that writes to Neptune AI platform."""

    metadata = None

    def __init__(
        self,
        config: DictConfig,
        mode: Literal["async", "sync", "offline", "read-only", "debug"] = "async",
        file_system: S3FileSystem | None = None,
        neptune_tags: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize NeptuneLogger.

        Args:
            config: Configuration dictionary containing logging settings
            mode: Neptune run mode. Defaults to "async"
            file_system: Optional S3 file system for artifact storage
            neptune_tags: Optional list of tags to add to the Neptune run
            **kwargs: Additional keyword arguments passed to neptune.init_run
        """
        self.config = config
        self.file_system = file_system
        self.save_training_plots = config.logging.get("save_training_plots", False)
        self.save_test_plots = config.logging.get("save_test_plots", True)
        self._timestep = 0

        # Process tags
        tags = self._process_tags(config, neptune_tags or [])

        # Initialize Neptune run
        self.run = neptune.init_run(
            project=config["logging"]["name"],
            tags=tags,
            mode=mode,
            **kwargs,
        )
        self.run["config"] = stringify_unsupported(config)

        # Store run ID if not offline
        if mode != "offline":
            self.run_id = self.run["sys/id"].fetch()
            NeptuneLogger.metadata = self.run_id

    def _process_tags(self, config: DictConfig, neptune_tags: list[str]) -> list[str]:
        """Process and combine configuration tags with neptune tags.

        Args:
            config: Configuration dictionary containing logging tags
            neptune_tags: List of additional tags to add

        Returns:
            Combined list of all tags

        Raises:
            ValueError: If config tags field is not None or a list
        """
        config_tags = config["logging"]["tags"]

        if isinstance(config_tags, listconfig.ListConfig):
            tags = list(config_tags)
        elif config_tags is None:
            tags = []
        else:
            raise ValueError("tags field must be None or list")

        return tags + neptune_tags

    def write(
        self,
        data: Dict[str, Union[float, plt.Figure]],
        label: str = "",
        timestep: int | None = None,
    ) -> None:
        """Write data to Neptune, handling both metrics and figures.

        Args:
            data: Dictionary of metrics (float) or figures (plt.Figure) to log
            label: Optional label prefix for the data
            timestep: Optional timestep for the data. If None, uses internal counter
        """
        try:
            self._timestep = timestep if timestep is not None else self._timestep + 1
            prefix = f"{label}/" if label else ""

            for key, value in data.items():
                if isinstance(value, plt.Figure):
                    self._log_figure(key, value, prefix)
                else:
                    self._log_metric(key, value, prefix)

        except Exception as e:
            log.error(f"Neptune Write Error: {e}")

    def _log_figure(self, key: str, figure: plt.Figure, prefix: str) -> None:
        """Log a matplotlib figure if conditions are met.

        Args:
            key: Key name for the figure
            figure: Matplotlib figure to log
            prefix: Prefix to add to the key name
        """
        should_save = ("test" in key and self.save_test_plots) or (
            "training" in key and self.save_training_plots
        )

        if should_save:
            self.run[f"{prefix}{key}"].append(figure, step=self._timestep)

    def _log_metric(self, key: str, value: float, prefix: str) -> None:
        """Log a numeric metric if it's not NaN.

        Args:
            key: Key name for the metric
            value: Numeric value to log
            prefix: Prefix to add to the key name
        """
        if not np.isnan(value):
            self.run[f"{prefix}{key}"].log(value, step=self._timestep, wait=False)

    def write_artifact(self, file_name: str) -> None:
        """Upload an artifact file to Neptune.

        Args:
            file_name: Path to the artifact file to upload
        """
        try:
            log.info("Saving checkpoint to Neptune")
            self.run[file_name].upload(file_name)
        except Exception as e:
            log.error(f"Neptune Write Artifact Error: {e}")

    def close(self) -> None:
        """Stop the Neptune run."""
        self.run.stop()

    def get_checkpoint(self, file_path: str) -> str | None:
        """Download a checkpoint from Neptune and return the local path.

        Args:
            file_path: Path in format 'run_id/checkpoint_name' to download

        Returns:
            Local path to the downloaded checkpoint, or None if download fails

        Raises:
            ValueError: If file_path is not in the format 'run_id/checkpoint_name'
        """
        try:
            parts = file_path.split("/")
            if len(parts) != 2:
                raise ValueError("File path must be in format 'run_id/checkpoint_name'")

            run = neptune.init_run(project=self.config["project"], with_id=parts[0])
            destination = f"{parts[0]}_{parts[1]}"
            run[f"checkpoints/{parts[1]}"].download(destination=destination)
            return destination
        except Exception as e:
            log.info(f"Unable to load checkpoint: {e}")
            return None


class TerminalLogger(Logger):
    """Simple logger that outputs to terminal/console."""

    def __init__(self, **kwargs: Any):
        """Initialize TerminalLogger.

        Args:
            **kwargs: Additional keyword arguments (currently unused)
        """
        log.info(">>> Terminal Logger")

    def write(
        self,
        data: Dict[str, Union[float, plt.Figure]],
        label: str = "",
        timestep: int | None = None,
    ) -> None:
        """Write metrics to terminal, ignoring figures.

        Args:
            data: Dictionary of metrics (float) or figures (plt.Figure) to log
            label: Optional label prefix (currently unused)
            timestep: Optional timestep to display with the metrics
        """
        metrics = []
        for key, value in data.items():
            if not isinstance(value, plt.Figure) and not np.isnan(value):
                metrics.append(f"{key}: {value:.5f}")

        if metrics:
            message = "\n".join(metrics)
            if timestep is not None:
                log.info(f"Step {timestep:.2e}:\n{message}")
            else:
                log.info(message)


def logger_factory(
    logger_type: str,
    config_dict: DictConfig,
    mode: Literal["async", "sync", "offline", "read-only", "debug"] = "async",
    file_system: S3FileSystem | None = None,
    **kwargs: Any,
) -> Logger:
    """Factory function to create logger instances.

    Args:
        logger_type: Type of logger to create ("neptune" or "terminal")
        config_dict: Configuration dictionary for the logger
        mode: Neptune run mode (only used for NeptuneLogger). Defaults to "async"
        file_system: Optional S3 file system (only used for NeptuneLogger)
        **kwargs: Additional keyword arguments passed to logger initialization

    Returns:
        Logger instance of the specified type

    Raises:
        ValueError: If logger_type is not "neptune" or "terminal"
    """
    if logger_type == "neptune":
        return NeptuneLogger(config_dict, mode=mode, file_system=file_system, **kwargs)
    elif logger_type == "terminal":
        return TerminalLogger(**kwargs)
    else:
        raise ValueError(
            f"Unsupported logger type: {logger_type}. Expected 'neptune' or 'terminal'."
        )


def get_logger_from_config(
    config: DictConfig, file_system: S3FileSystem | None = None, **kwargs: Any
) -> Logger:
    """Create a logger based on configuration settings.

    Args:
        config: Configuration dictionary containing logging settings
        file_system: Optional S3 file system (only used for NeptuneLogger)
        **kwargs: Additional keyword arguments passed to logger initialization

    Returns:
        Logger instance configured according to the config settings
    """
    return logger_factory(
        logger_type=config.logging.type,
        config_dict=config,
        mode=config.logging.mode,
        file_system=file_system,
        **kwargs,
    )
