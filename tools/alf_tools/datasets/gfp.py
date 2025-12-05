import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from alf_core import BaseDataset, Candidate, LabeledCandidates

logger = logging.getLogger("alf-tools")

DATAPATH = "alf_tools/datasets/data/"
FILENAME = "gfp_dataset.csv"
URL = "https://raw.githubusercontent.com/dhbrookes/CbAS/master/data/gfp_data.csv"


class GFP(BaseDataset):
    """GFP dataset class."""

    def __init__(self, name: str, modality: str, seed: int, split_config: dict[str, Any]):
        """Initialize the GFP dataset.

        Args:
            name: The name of the dataset.
            modality: The modality of the dataset.
            seed: The seed for the dataset.
            split_config: The split configuration for the dataset.
        """
        super().__init__(name, modality, seed, split_config)
        self.setup()

    def load_dataset(self) -> LabeledCandidates:
        """Load GFP dataset from local file or download from URL if not present.
        Clean dataset and return as HF dataset.

        Returns:
            LabeledCandidates: A LabeledCandidates object containing the GFP data.
        """
        if not os.path.exists(os.path.join(DATAPATH, FILENAME)):
            os.makedirs(DATAPATH, exist_ok=True)
            response = requests.get(URL)
            if response.status_code == 200:
                with open(os.path.join(DATAPATH, FILENAME), "wb") as file:
                    file.write(response.content)
                logger.info("GFP dataset downloaded successfully.")
            else:
                raise ValueError(f"Failed to download GFP dataset. Status code: {response.status_code}")

        gfp_dataset = pd.read_csv(os.path.join(DATAPATH, FILENAME))
        # Note, here we are only using the first 1000 rows of the dataset,
        # otherwise the candidate pool is too large and becomes compute intensive
        data = list(gfp_dataset["nucSequence"])[:1000]
        labels = np.array(gfp_dataset["medianBrightness"].values)[:1000]
        return LabeledCandidates(
            candidates=[Candidate(data=data, modality=self.modality) for data in data],
            labels=labels,
        )
