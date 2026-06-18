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

r"""Benchmark: compare surrogate models on a dataset.

Runs the same active-learning loop (greedy acquisition, fixed dataset/splits per seed)
with three surrogates — CNN, GP, and ESM-2 (small 8M, linear head) — averaged over
seeds. Produces:

  * a predictive-quality curve (test Spearman vs acquisition round),
  * a best-found-so-far curve,
  * a printed summary table + summary.csv.

A taster for surrogate benchmarking; the deferred alf_benchmark layer generalises it.

Example:
    uv run --group benchmark python benchmark_examples/benchmarking_surrogates.py \\
        --dataset gfp --num-rounds 5 --batch-size 50 --num-seeds 3
"""

from pathlib import Path

import bench
import matplotlib.pyplot as plt


def main() -> None:
    """Run the surrogate comparison and write plots + summary."""
    parser = bench.base_arg_parser(
        description=__doc__, default_output="benchmark_examples/outputs/surrogates"
    )
    parser.add_argument(
        "--esm-model-id",
        type=str,
        default=bench.DEFAULT_ESM_MODEL_ID,
        help=f"ESM-2 HuggingFace checkpoint (default: {bench.DEFAULT_ESM_MODEL_ID}).",
    )
    args = parser.parse_args()
    bench.configure_logging()
    out = Path(args.output_dir)

    surrogates = {
        "cnn": bench.build_cnn,
        "gp": bench.build_gp,
        "esm2": lambda seed, epochs: bench.build_esm(seed, epochs, args.esm_model_id),
    }

    results: dict[str, list] = {}
    for model_name, build_model in surrogates.items():
        per_seed = []
        for seed in range(args.num_seeds):
            bench.log.info("Running surrogate=%s seed=%d ...", model_name, seed)
            per_seed.append(
                bench.run_al_experiment(
                    dataset_name=args.dataset,
                    model_name=model_name,
                    build_model=build_model,
                    acquisition="greedy",
                    seed=seed,
                    num_rounds=args.num_rounds,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    workdir=out / "runs" / f"{model_name}_seed{seed}",
                )
            )
        results[model_name] = per_seed

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bench.plot_metric(
        axes[0],
        results,
        "surrogate/test_spearman",
        title=f"Surrogate predictive quality ({args.dataset})",
        ylabel="test Spearman",
    )
    bench.plot_metric(
        axes[1],
        results,
        "best_found_so_far",
        title=f"Best found so far ({args.dataset})",
        ylabel="best label acquired",
    )
    bench.save_figure(fig, out / "surrogates_comparison.png")

    summary = bench.summarise_final(
        results,
        {"test_spearman": "surrogate/test_spearman", "best_found": "best_found_so_far"},
    )
    print("\n=== Surrogate comparison (final round, mean ± std over seeds) ===")
    print(summary.to_string())
    summary.to_csv(out / "summary.csv")
    bench.log.info("Saved summary: %s", out / "summary.csv")


if __name__ == "__main__":
    main()
