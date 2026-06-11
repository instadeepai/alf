# Dataset card: FLIP

**Registry name:** `flip` · **Modality:** sequence · **Problem type:** regression

## Summary

FLIP (*Fitness Landscape Inference for Proteins*) is a collection of protein
fitness-prediction tasks with **designed train/test splits** that probe
generalisation (e.g. low-vs-high, mutational-distance, sampled). In ALF it is an
*extended-tier* benchmark problem: it requires a data download and is therefore
not part of the committed CPU baseline.

## Provenance

- **Benchmark:** Dallago et al., *"FLIP: Benchmark tasks in fitness landscape
  inference for proteins"*, NeurIPS 2021 Datasets & Benchmarks.
- **As consumed by ALF:** selected via `FLIPConfig` with
  `flip_dataset ∈ {aav, gb1, meltome, scl, sav}` and a `flip_split`. See
  [`tools/alf_tools/datasets/flip.py`](../../tools/alf_tools/datasets/flip.py)
  for the exact source, download behaviour, and split semantics.

## Split

FLIP ships **predefined splits** per task (the point of the benchmark). The
`flip_split` field selects one; do not re-randomise it if you want results
comparable to the FLIP literature. The `seed` still controls any ALF-side
shuffling of the acquisition pool.

## License

See the FLIP project for dataset licences and citation requirements; confirm
terms before redistribution.

## Known limitations / leakage notes

- Splits are **designed for distribution shift** — choosing the wrong split (or
  re-randomising) silently changes the difficulty and breaks comparability.
- Task scales vary widely (Meltome is large); plan compute accordingly. This is
  why FLIP is documented as an optional tier rather than a committed baseline.
