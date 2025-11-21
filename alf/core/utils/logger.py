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
from typing import Any

log = logging.getLogger("alf-core")


class Logger(abc.ABC):
    """Abstract base class for loggers that can write data and artifacts."""

    @abc.abstractmethod
    def write(self, data: dict[str, Any], label: str = "", timestep: int | None = None) -> None:
        """Write data to the logger destination.

        Args:
            data: Dictionary of data to write
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
        data: dict[str, float],
        label: str = "",
        timestep: int | None = None,
    ) -> None:
        """Write metrics to terminal.

        Args:
            data: Dictionary of metrics to log
            label: Optional label prefix (currently unused)
            timestep: Optional timestep to display with the metrics
        """
        metrics = [f"{key}: {value:.5f}" for key, value in data.items()]

        if metrics:
            message = "\n".join(metrics)
            if timestep is not None:
                log.info(f"Step {timestep:.2e}:\n{message}")
            else:
                log.info(message)
