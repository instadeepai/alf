# Copyright 2026 InstaDeep Ltd. All rights reserved.
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

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from alf_core import BaseDataset, BaseDatasetConfig, Candidate, LabelledCandidates

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"
FILENAME = "gfp_dataset.csv"
URL = "https://raw.githubusercontent.com/dhbrookes/CbAS/master/data/gfp_data.csv"


class GFP(BaseDataset):
    """GFP dataset class."""

    def __init__(self, config: BaseDatasetConfig):
        """Initialize the GFP dataset.

        Args:
            config: Configuration for the GFP dataset.
        """
        super().__init__(config)
        self.setup()

    def load_dataset(self) -> LabelledCandidates:
        """Load GFP dataset from local file or download from URL if not present.
        Clean dataset and return as HF dataset.

        Returns:
            A LabelledCandidates object containing the GFP data.

        Raises:
            FileNotFoundError: If the GFP dataset file is not found.
        """
        filepath = DATAPATH / FILENAME
        if not filepath.exists():
            DATAPATH.mkdir(parents=True, exist_ok=True)
            response = requests.get(URL)
            if response.status_code == 200:
                with open(filepath, "wb") as file:
                    file.write(response.content)
                logger.info("GFP dataset downloaded successfully.")
            else:
                raise FileNotFoundError(
                    f"Failed to download GFP dataset. Status code: {response.status_code}"
                )

        gfp_dataset = pd.read_csv(filepath)
        # Note, here we are only using the first 1000 rows of the dataset,
        # otherwise the candidate pool is too large and becomes compute intensive
        data = list(gfp_dataset["nucSequence"])[:1000]
        labels = np.array(gfp_dataset["medianBrightness"].values)[:1000]
        return LabelledCandidates(
            candidates=[Candidate(data=data, modality=self.modality) for data in data],
            labels=labels,
        )
