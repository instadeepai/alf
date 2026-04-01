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

# ruff: noqa: E402
import beartype.claw

beartype.claw.beartype_this_package()

from alf_core.dataclasses.candidate import Candidate, Modality
from alf_core.dataclasses.labelled_candidates import LabelledCandidates
from alf_core.dataclasses.predictions import Predictions
from alf_core.dataclasses.results import Results
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.dataclasses.state import State
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
