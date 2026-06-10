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

"""The ``alf-bench`` command-line interface.

Subcommands let the wider community run, aggregate, plot, and inspect benchmarks
declaratively, without writing Python:

- ``alf-bench run <config.yaml>`` — run a sweep from a YAML config.
- ``alf-bench aggregate <dir>`` — print a leaderboard (optionally write aggregate CSV).
- ``alf-bench plot <dir>`` — save active-learning curve figures.
- ``alf-bench list`` — show registered components discovered via entry points.
"""

import argparse
import logging
from pathlib import Path

from alf_benchmark.registry import GROUPS, default_registry
from alf_benchmark.results import BenchmarkResults
from alf_benchmark.runner import run_benchmark
from alf_benchmark.yaml_loader import load_run_config

logger = logging.getLogger("alf-benchmark")


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a sweep described by a YAML config.

    Args:
        args: Parsed arguments with ``config``.

    Returns:
        Exit code (non-zero if any replication failed).
    """
    run_config = load_run_config(args.config)
    manifests = run_benchmark(run_config)
    completed = sum(m.status == "completed" for m in manifests)
    failed = len(manifests) - completed
    print(
        f"Ran {len(manifests)} replication(s): {completed} completed, {failed} failed. "
        f"Results in {Path(run_config.output_dir).resolve()}"
    )
    return 1 if failed else 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    """Print a leaderboard and optionally write the aggregate table.

    Args:
        args: Parsed arguments with ``dir`` and optional ``output``.

    Returns:
        Exit code (always 0).
    """
    results = BenchmarkResults.from_dir(args.dir)
    board = results.leaderboard()
    if board.empty:
        print("No completed results found.")
        return 0
    print(board.to_string(index=False))
    if args.output:
        results.aggregate().to_csv(args.output, index=False)
        print(f"\nWrote aggregate table to {args.output}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    """Save an active-learning curve figure per problem.

    Args:
        args: Parsed arguments with ``dir``, optional ``metric`` and ``out_dir``.

    Returns:
        Exit code (0 on success, 1 if there is nothing to plot).
    """
    # Imported lazily so non-plot subcommands don't pull in matplotlib.
    from alf_benchmark.plotting import plot_curves  # noqa: PLC0415

    results = BenchmarkResults.from_dir(args.dir)
    if results.long.empty:
        print("No completed results found.")
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = sorted(results.long["problem"].unique())
    for problem in problems:
        metric = args.metric or results.primary_metrics.get(problem, "optimizer/regret")
        try:
            figure = plot_curves(results, metric=metric, problem=problem)
        except ValueError as error:
            logger.warning("Skipping '%s': %s", problem, error)
            continue
        safe_metric = metric.replace("/", "_")
        path = out_dir / f"{problem}_{safe_metric}.png"
        figure.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """List components discovered under each entry-point group.

    Args:
        args: Parsed arguments (unused).

    Returns:
        Exit code (always 0).
    """
    registry = default_registry()
    for group in GROUPS:
        names = registry.names(group)
        print(f"{group}:")
        for name in names:
            print(f"  {name}")
        if not names:
            print("  (none)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``alf-bench`` argument parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(prog="alf-bench", description="ALF benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a sweep from a YAML config")
    run_parser.add_argument("config", help="path to a YAML run config")
    run_parser.set_defaults(func=_cmd_run)

    aggregate_parser = subparsers.add_parser("aggregate", help="print a leaderboard")
    aggregate_parser.add_argument("dir", help="run output directory")
    aggregate_parser.add_argument(
        "--output", help="optional path to write the aggregate table as CSV"
    )
    aggregate_parser.set_defaults(func=_cmd_aggregate)

    plot_parser = subparsers.add_parser("plot", help="save AL curve figures")
    plot_parser.add_argument("dir", help="run output directory")
    plot_parser.add_argument("--metric", help="metric to plot (default: primary metric)")
    plot_parser.add_argument("--out-dir", dest="out_dir", help="directory to write figures to")
    plot_parser.set_defaults(func=_cmd_plot)

    list_parser = subparsers.add_parser("list", help="list registered components")
    list_parser.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``alf-bench`` console script.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))
