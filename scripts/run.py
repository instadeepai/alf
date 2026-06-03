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

"""Hydra entry point for ALF experiments."""

import logging
from pathlib import Path

import factory
import hydra
from alf_core.surrogate.surrogate import Surrogate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run an ALF experiment from a Hydra config.

    Args:
        cfg: Composed Hydra DictConfig (model, dataset, optimizer, oracle, task, experiment).
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("Output directory: %s", output_dir)
    log.info(
        "Phase: %s | Experiment: %s | Seed: %s",
        cfg.experiment.phase,
        cfg.experiment.name,
        cfg.experiment.seed,
    )

    dataset = factory.build_dataset(cfg)
    surrogate = Surrogate(model=factory.build_model(cfg))
    optimizer = factory.build_optimizer(cfg)
    oracle = factory.build_oracle(cfg, dataset)
    loggers = factory.build_state_loggers(cfg, output_dir)
    task = factory.build_task(cfg)

    dataset.setup()
    state = task.setup(dataset=dataset, surrogate=surrogate)
    task.run(state=state, state_loggers=loggers, optimizer=optimizer, oracle=oracle)


if __name__ == "__main__":
    main()
