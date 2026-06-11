# Dataset card: GFP

**Registry name:** `gfp` · **Modality:** sequence · **Problem type:** regression

## Summary

A fluorescence fitness landscape of the green fluorescent protein (avGFP): each
candidate is a sequence and the label is its measured brightness. Used here as
the bundled, no-download, CPU-friendly design problem in `alf-protein-v1`.

## Provenance

- **Source file:** `gfp_data.csv` from the CbAS repository
  (<https://raw.githubusercontent.com/dhbrookes/CbAS/master/data/gfp_data.csv>),
  auto-downloaded and cached under `tools/alf_tools/datasets/data/` on first use.
- **Underlying data:** the avGFP local fitness landscape of Sarkisyan et al.,
  *"Local fitness landscape of the green fluorescent protein"*, Nature 2016
  (deep mutational scanning of avGFP).
- **As consumed by ALF:** the loader reads the `nucSequence` column as the
  candidate and `medianBrightness` as the label, and **uses the first 1000 rows
  only** (to keep the candidate pool small enough for fast CPU runs). See
  [`tools/alf_tools/datasets/gfp.py`](../../tools/alf_tools/datasets/gfp.py).

## Split

- Random split controlled by the config `seed`, via the standard
  `BaseDataset` splitter: `train_ratio` / `validation_frac` / `test_ratio`.
- `alf-protein-v1` pins `train_ratio=0.1`, `validation_frac=0.2`,
  `test_ratio=0.2` (the remainder forms the acquisition candidate pool) over
  seeds `[0, 1, 2]`. With 1000 rows this yields train≈80, val≈20, test≈200,
  pool≈700. The initial train split is kept deliberately small so the pool (not
  the model's training data) holds the optimum, and acquisition drives the result.

## License

The avGFP measurements are from a published academic study (Sarkisyan et al.,
2016); the CSV is redistributed via the CbAS repository. Confirm the upstream
licence terms before any redistribution beyond research use.

## Known limitations / leakage notes

- **Truncation:** only the first 1000 rows are used, so this is a *subset* of the
  full landscape — results are not comparable to runs over the complete dataset.
- **Nucleotide sequences:** candidates are nucleotide (`nucSequence`) strings,
  not amino-acid sequences; featurisers must treat them as such.
- **Random split:** the offline candidate pool is a random hold-out, so there is
  no designed train/test distribution shift; the split is reproducible only via
  the fixed `seed`.
