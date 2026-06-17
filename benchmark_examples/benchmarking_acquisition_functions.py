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

r"""Benchmark: compare acquisition functions on a dataset (fixed GP surrogate).

Runs the same active-learning loop with a fixed GP surrogate (which provides the
uncertainty UCB/EI need) and four acquisition functions — greedy, UCB, EI, core_set —
averaged over seeds (shared splits per seed). Produces:

  * split-relative headline curves: regret and top-K recall vs acquisition round,
  * a best-found-so-far curve,
  * a printed summary table + summary.csv.

Note: core_set is a pure-diversity (exploration) baseline — it selects diverse points
and ignores the surrogate's predictions, so it is expected to trail on best-found.

Example:
    uv run --group benchmark python benchmark_examples/benchmarking_acquisition_functions.py \\
        --dataset gfp --num-rounds 5 --batch-size 50 --num-seeds 3
"""

from pathlib import Path

import bench
import matplotlib.pyplot as plt

ACQUISITIONS = ["greedy", "ucb", "ei", "core_set"]


def main() -> None:
    """Run the acquisition-function comparison and write plots + summary."""
    parser = bench.base_arg_parser(
        description=__doc__, default_output="benchmark_examples/outputs/acquisition_functions"
    )
    args = parser.parse_args()
    bench.configure_logging()
    out = Path(args.output_dir)

    results: dict[str, list] = {}
    for acquisition in ACQUISITIONS:
        per_seed = []
        for seed in range(args.num_seeds):
            bench.log.info("Running acquisition=%s seed=%d ...", acquisition, seed)
            per_seed.append(
                bench.run_al_experiment(
                    dataset_name=args.dataset,
                    model_name="gp",  # GP provides the uncertainty UCB/EI need
                    build_model=bench.build_gp,
                    acquisition=acquisition,
                    seed=seed,
                    num_rounds=args.num_rounds,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    workdir=out / "runs" / f"{acquisition}_seed{seed}",
                )
            )
        results[acquisition] = per_seed

    recall_col = bench.find_recall_column(results["greedy"][0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bench.plot_metric(
        axes[0],
        results,
        "optimizer/regret",
        title=f"Regret ({args.dataset})",
        ylabel="regret (lower is better)",
    )
    if recall_col is not None:
        bench.plot_metric(
            axes[1],
            results,
            recall_col,
            title=f"Top-K recall ({args.dataset})",
            ylabel=recall_col.split("/")[-1],
        )
    else:
        axes[1].set_visible(False)
    bench.plot_metric(
        axes[2],
        results,
        "best_found_so_far",
        title=f"Best found so far ({args.dataset})",
        ylabel="best label acquired",
    )
    bench.save_figure(fig, out / "acquisition_functions_comparison.png")

    columns = {"regret": "optimizer/regret", "best_found": "best_found_so_far"}
    if recall_col is not None:
        columns["recall"] = recall_col
    summary = bench.summarise_final(results, columns)
    print("\n=== Acquisition comparison (final round, mean ± std over seeds) ===")
    print(summary.to_string())
    summary.to_csv(out / "summary.csv")
    bench.log.info("Saved summary: %s", out / "summary.csv")


if __name__ == "__main__":
    main()
