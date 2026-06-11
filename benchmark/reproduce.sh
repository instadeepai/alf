#!/usr/bin/env bash
# Reproduce the alf-protein-v1 reference leaderboard.
#
# CPU-friendly and offline: uses the bundled GFP dataset (no GPU, no HF token,
# no download). Run from anywhere; paths are resolved relative to the repo root.
#
#   bash benchmark/reproduce.sh
#
# Compare the regenerated benchmark/reference_results/LEADERBOARD.md against the
# committed one — the mean regret per method should match within the CI.
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

SUITE="benchmark/alf_benchmark/suites/alf_protein_v1.yaml"
RUNS="runs/alf_protein_v1"
REF="benchmark/reference_results"

uv run alf-bench run "$SUITE"
uv run alf-bench aggregate "$RUNS" \
    --markdown "$REF/LEADERBOARD.md" \
    --output "$REF/aggregate.csv"

echo "Reproduced. See $REF/LEADERBOARD.md"
