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

"""End-to-end tests for the alf-bench CLI over synthetic components."""

import matplotlib
from alf_benchmark.cli import main

matplotlib.use("Agg")

_RUN_YAML = """
suite_name: cli_test
suite_version: "0.1.0"
output_dir: "{output_dir}"
problems:
  - name: dummy_problem
    primary_metric: optimizer/regret
    seeds: [0, 1]
    dataset:
      name: dummy
      config:
        name: dummy
        modality: sequence
        train_ratio: 0.4
        validation_frac: 0.2
        test_ratio: 0.2
        problem_type: regression
        num_samples: 400
methods:
  - name: m_dummy
    family: design
    surrogate: {{name: dummy}}
    acquisition: {{name: dummy_acq}}
    search: {{name: dataset_search}}
    task: {{num_acq_rounds: 2, acq_batch_size: 10}}
"""


def test_cli_list_shows_real_components(capsys):
    """`alf-bench list` prints groups and discovered component names."""
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "alf.models:" in out
    assert "cnn" in out
    assert "gfp" in out


def test_cli_run_aggregate_plot(dummies_in_default_registry, tmp_path, capsys):
    """`run` produces outputs that `aggregate` and `plot` then consume."""
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(_RUN_YAML.format(output_dir=runs_dir.as_posix()))

    # run
    assert main(["run", str(config_path)]) == 0
    rep_dir = runs_dir / "dummy_problem" / "m_dummy" / "seed=0"
    assert (rep_dir / "metrics.csv").exists()
    assert (rep_dir / "manifest.json").exists()

    # aggregate (+ markdown leaderboard)
    markdown = tmp_path / "LEADERBOARD.md"
    assert main(["aggregate", str(runs_dir), "--markdown", str(markdown)]) == 0
    assert "m_dummy" in capsys.readouterr().out
    assert markdown.exists()
    text = markdown.read_text()
    assert "# Leaderboard" in text
    assert "m_dummy" in text

    # plot
    assert main(["plot", str(runs_dir)]) == 0
    pngs = list((runs_dir / "plots").glob("*.png"))
    assert pngs


def test_cli_aggregate_empty_dir(tmp_path, capsys):
    """`aggregate` on an empty directory reports nothing found, exits 0."""
    assert main(["aggregate", str(tmp_path)]) == 0
    assert "No completed results" in capsys.readouterr().out
