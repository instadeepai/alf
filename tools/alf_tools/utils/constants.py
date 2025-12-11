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

import os

PROTEIN_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
HF_DATASETS_REPOSITORY_NAME = "InstaDeepAI/alfred"
HF_DATASETS_REPOSITORY_URL = f"hf://datasets/{HF_DATASETS_REPOSITORY_NAME}"
BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
