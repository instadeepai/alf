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

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml


class FileHandler:
    """File handler for local storage.

    Provides a consistent interface for local file operations.
    """

    def __init__(self, base_path: str = "./") -> None:
        """Initialize FileHandler.

        Args:
            base_path: Base directory path for file operations. Defaults to "./"
        """
        self.base_path = base_path

    def _get_full_path(self, path: str) -> str:
        """Get the full path for the given relative path.

        Args:
            path: Relative path to the file

        Returns:
            Full path to the file
        """
        return os.path.join(self.base_path, path)

    def _ensure_local_dir(self, path: str) -> None:
        """Ensure local directory exists for the given path.

        Args:
            path: Path to the directory
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def _open_file(self, path: str, mode: str = "r", **kwargs):
        """Open file with local filesystem.

        Args:
            path: Path to the file
            mode: Mode to open the file in
            **kwargs: Additional arguments for the open function

        Returns:
            File handle
        """
        full_path = self._get_full_path(path)
        return open(full_path, mode, **kwargs)

    def open(self, path: str, *args: Any, **kwargs: Any):
        """Open file with local filesystem.

        This is a public interface for the _open_file method.

        Args:
            path: Path to the file
            *args: Additional positional arguments for the open function
            **kwargs: Additional keyword arguments for the open function

        Returns:
            File handle
        """
        return self._open_file(path, *args, **kwargs)

    def read_numpy(self, path: str) -> np.ndarray:
        """Load numpy array from file.

        Args:
            path: Path to the numpy file

        Returns:
            Loaded numpy array
        """
        full_path = self._get_full_path(path)
        return np.load(full_path, allow_pickle=True)

    def save_numpy(self, path: str, array: np.ndarray) -> None:
        """Save numpy array to file.

        Args:
            path: Path where the array should be saved
            array: Numpy array to save
        """
        full_path = self._get_full_path(path)
        self._ensure_local_dir(full_path)
        np.save(full_path, array)

    def read_text(self, path: str) -> list[str]:
        """Read text file and return lines.

        Args:
            path: Path to the text file

        Returns:
            List of lines from the file
        """
        with self._open_file(path, "r") as f:
            return f.readlines()

    def save_text(self, path: str, lines: list[str]) -> None:
        """Save lines to text file.

        Args:
            path: Path where the text should be saved
            lines: List of lines to write to the file
        """
        full_path = self._get_full_path(path)
        self._ensure_local_dir(full_path)

        with self._open_file(path, "w") as f:
            for line in lines:
                f.write(line)

    def read_json(self, path: str) -> dict:
        """Load JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Dictionary from the JSON file
        """
        with self._open_file(path, "r") as f:
            return json.load(f)

    def save_json(self, path: str, data: dict) -> None:
        """Save dictionary to JSON file.

        Args:
            path: Path where the JSON should be saved
            data: Dictionary to save as JSON
        """
        full_path = self._get_full_path(path)
        self._ensure_local_dir(full_path)

        with self._open_file(path, "w") as f:
            json.dump(data, f, indent=4)

    def read_yaml(self, path: str) -> dict:
        """Load YAML file safely.

        Args:
            path: Path to the YAML file

        Returns:
            Dictionary from the YAML file
        """
        with self._open_file(path, "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def save_yaml(self, path: str, data: dict) -> None:
        """Save dictionary to YAML file.

        Args:
            path: Path where the YAML should be saved
            data: Dictionary to save as YAML
        """
        full_path = self._get_full_path(path)
        self._ensure_local_dir(full_path)

        with self._open_file(path, "w") as f:
            yaml.dump(data, f)

    def read_csv(self, path: str, header: str = "infer") -> pd.DataFrame:
        """Load CSV file as pandas DataFrame.

        Args:
            path: Path to the CSV file
            header: Row to use for column names. Defaults to "infer"

        Returns:
            Pandas DataFrame from the CSV file
        """
        full_path = self._get_full_path(path)
        return pd.read_csv(full_path, header=header)

    def save_csv(
        self, path: str, df: pd.DataFrame, header: bool = True, index: bool = False
    ) -> None:
        """Save pandas DataFrame to CSV file.

        Args:
            path: Path where the CSV should be saved
            df: DataFrame to save
            header: Whether to include headers. Defaults to True
            index: Whether to include index. Defaults to False
        """
        full_path = self._get_full_path(path)
        self._ensure_local_dir(full_path)
        df.to_csv(full_path, index=index, header=header)

    def listdir(self, path: str) -> list[str]:
        """List files in directory.

        Args:
            path: Path to the directory

        Returns:
            List of filenames in the directory
        """
        full_path = self._get_full_path(path)
        return os.listdir(full_path)

    def isfile(self, path: str) -> bool:
        """Check if path is a file.

        Args:
            path: Path to check

        Returns:
            True if path is a file, False otherwise
        """
        full_path = self._get_full_path(path)
        return os.path.isfile(full_path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directories.

        Args:
            path: Directory path to create
            exist_ok: If True, don't raise error if directory exists
        """
        full_path = self._get_full_path(path)
        os.makedirs(full_path, exist_ok=exist_ok)


# Global file handler instance for convenience
input_handler = FileHandler()
