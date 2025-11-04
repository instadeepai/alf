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
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from s3fs.core import S3FileSystem


class FileHandler:
    """Unified file handler for local and S3 storage.

    Provides a consistent interface for file operations whether using local filesystem
    or S3 storage. Automatically handles path resolution and file system abstraction.
    """

    def __init__(
        self, s3_endpoint: Optional[str] = None, bucket: Optional[str] = "input"
    ) -> None:
        """Initialize FileHandler.

        Args:
            s3_endpoint: S3 endpoint URL. If None, uses local filesystem
            bucket: Bucket type - "input" or "output". Only used with S3.
        """
        self.s3_endpoint = s3_endpoint
        self.bucket = bucket

        if s3_endpoint:
            self.s3 = S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint})
            if bucket == "input":
                self.bucket_path = os.environ["AICHOR_INPUT_PATH"]
            elif bucket == "output":
                self.bucket_path = os.environ["AICHOR_OUTPUT_PATH"]
            else:
                raise ValueError("bucket must be 'input' or 'output'")
        else:
            self.bucket_path = "./"

    def _get_full_path(self, path: str) -> str:
        """Get the full path for the given relative path."""
        return os.path.join(self.bucket_path, path)

    def _ensure_local_dir(self, path: str) -> None:
        """Ensure local directory exists for the given path."""
        if not self.s3_endpoint:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

    def _open_file(self, path: str, mode: str = "r", **kwargs):
        """Open file with appropriate handler (S3 or local)."""
        full_path = self._get_full_path(path)
        if self.s3_endpoint:
            return self.s3.open(full_path, mode, **kwargs)
        else:
            return open(full_path, mode, **kwargs)

    def open(self, path: str, *args: Any, **kwargs: Any):
        """Open file with appropriate handler (S3 or local).

        This is a public interface for the _open_file method.
        """
        return self._open_file(path, *args, **kwargs)

    def read_numpy(self, path: str) -> np.ndarray:
        """Load numpy array from file.

        Args:
            path: Path to the numpy file

        Returns:
            Loaded numpy array
        """
        if self.s3_endpoint:
            with self._open_file(path, "rb") as f:
                return np.load(f, allow_pickle=True)
        else:
            return np.load(path, allow_pickle=True)

    def save_numpy(self, path: str, array: np.ndarray) -> None:
        """Save numpy array to file.

        Args:
            path: Path where the array should be saved
            array: Numpy array to save
        """
        self._ensure_local_dir(path)

        if self.s3_endpoint:
            with self._open_file(path, "wb") as f:
                f.write(pickle.dumps(array))
        else:
            np.save(path, array)

    def read_text(self, path: str) -> List[str]:
        """Read text file and return lines.

        Args:
            path: Path to the text file

        Returns:
            List of lines from the file
        """
        with self._open_file(path, "r") as f:
            return f.readlines()

    def save_text(self, path: str, lines: List[str]) -> None:
        """Save lines to text file.

        Args:
            path: Path where the text should be saved
            lines: List of lines to write to the file
        """
        self._ensure_local_dir(path)

        with self._open_file(path, "w") as f:
            for line in lines:
                f.write(line)

    def read_json(self, path: str) -> Dict:
        """Load JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Dictionary from the JSON file
        """
        with self._open_file(path, "r") as f:
            return json.load(f)

    def save_json(self, path: str, data: Dict) -> None:
        """Save dictionary to JSON file.

        Args:
            path: Path where the JSON should be saved
            data: Dictionary to save as JSON
        """
        self._ensure_local_dir(path)

        with self._open_file(path, "w") as f:
            json.dump(data, f, indent=4)

    def read_yaml(self, path: str) -> Dict:
        """Load YAML file safely.

        Args:
            path: Path to the YAML file

        Returns:
            Dictionary from the YAML file
        """
        with self._open_file(path, "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def save_yaml(self, path: str, data: Dict) -> None:
        """Save dictionary to YAML file.

        Args:
            path: Path where the YAML should be saved
            data: Dictionary to save as YAML
        """
        self._ensure_local_dir(path)

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
        if self.s3_endpoint:
            assert "FSSPEC_S3_ENDPOINT_URL" in os.environ

        return pd.read_csv(self._get_full_path(path), header=header)

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
        if self.s3_endpoint:
            assert "FSSPEC_S3_ENDPOINT_URL" in os.environ
        else:
            self._ensure_local_dir(path)

        df.to_csv(self._get_full_path(path), index=index, header=header)

    def listdir(self, path: str) -> List[str]:
        """List files in directory.

        Args:
            path: Path to the directory

        Returns:
            List of filenames in the directory
        """
        if self.s3_endpoint:
            full_path = self._get_full_path(path)
            files = list(self.s3.ls(full_path))
            # Extract just the filenames from full paths
            return [os.path.basename(file) for file in files]
        else:
            return os.listdir(path)

    def isfile(self, path: str) -> bool:
        """Check if path is a file.

        Args:
            path: Path to check

        Returns:
            True if path is a file, False otherwise
        """
        if self.s3_endpoint:
            return self.s3.isfile(self._get_full_path(path))
        else:
            return os.path.isfile(path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directories.

        Args:
            path: Directory path to create
            exist_ok: If True, don't raise error if directory exists
        """
        if not self.s3_endpoint:
            os.makedirs(path, exist_ok=exist_ok)

    def download(
        self, remote_path: str, local_path: str, recursive: bool = False, **kwargs: Any
    ) -> None:
        """Download file from S3 to local.

        Args:
            remote_path: S3 path to download from
            local_path: Local path to download to
            recursive: Whether to download recursively
            **kwargs: Additional arguments for S3 download
        """
        if not self.s3_endpoint:
            raise ValueError("download() only works with S3 endpoint")
        self.s3.download(
            self._get_full_path(remote_path), local_path, recursive=recursive, **kwargs
        )

    def upload(
        self, local_path: str, remote_path: str, recursive: bool = False, **kwargs: Any
    ) -> None:
        """Upload file from local to S3.

        Args:
            local_path: Local path to upload from
            remote_path: S3 path to upload to
            recursive: Whether to upload recursively
            **kwargs: Additional arguments for S3 upload
        """
        if not self.s3_endpoint:
            raise ValueError("upload() only works with S3 endpoint")
        self.s3.put(
            local_path, self._get_full_path(remote_path), recursive=recursive, **kwargs
        )


# Global file handler instance for convenience
input_handler = FileHandler(os.environ.get("S3_ENDPOINT"), bucket="input")
