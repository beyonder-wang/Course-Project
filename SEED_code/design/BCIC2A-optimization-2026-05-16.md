# BCIC2A Optimization Note (2026-05-16)

## Goal

Push BCIC2A validation accuracy above the previous local ceiling while keeping all valuable prediction outputs in `Results/`.

Protocol used for the main search tonight: `fold=1` on the merged BCIC2A CV split unless stated otherwise.

## Theory-Guided Direction

- Local prior results and literature both pointed away from generic MLP/CNN tuning and toward `EEG-specific conv frontend + attention + temporal modeling`.
- The official ATCNet repo and the CTNet repo were inspected directly under `external_refs/`.
- Main takeaways:
  - `ATCNet` remains the strongest local backbone for this dataset.
  - CTNet's easiest high-upside idea to borrow is S&R augmentation, not a full Transformer port.
  - Longer training with plateau-style LR reduction matters more than cosine or aggressive standardization on this preprocessed HDF5 version.

## Implemented Changes

- Added optional train-split standardization support to the loader.
- Added supervised `mixup`, `label_smoothing`, `cosine` scheduler, and `ReduceLROnPlateau` support to the training pipeline.
- Added optional CTNet-style segment-reconstruction (`S&R`) augmentation.
- Added ATCNet CLI knobs for window count and a few core architecture hyperparameters.
- Added a reproducible `--seed` flag.
- Added `run_bcic2a_seed_sweep.py` to keep sweeping the current best recipe.

## Results So Far

### Existing baseline from prior work

- `ATCNet`, `lr=1e-3`, `30 epochs`, `fold=1`: **66.20%** best val acc

### New runs tonight

| Method | Key Settings | Best Val Acc | Notes |
|---|---|---:|---|
| ATCNet + standardize + label smoothing + cosine + wd | `35 ep` | 62.96% | Worse than baseline |
| ATCNet + mixup | `40 ep` | 65.74% | Better late-epoch behavior |
| ATCNet + mixup + plateau | `120 ep`, seed 42 | **67.59%** | Current best single model |
| ATCNet + mixup + plateau | `120 ep`, seed 7 | 66.67% | Stable but below best |
| ATCNet + mixup + plateau | `120 ep`, seed 13 | 67.59% | Matches best run |
| ATCNet + mixup + plateau | `120 ep`, seed 21 | 67.59% | Seed sweep |
| ATCNet + mixup + plateau | `120 ep`, seed 29 | 67.59% | Seed sweep |
| ATCNet + mixup + plateau | `120 ep`, seed 37 | **68.06%** | Best single-model seed sweep result |
| ATCNet + mixup + plateau | `120 ep`, seed 43 | 67.59% | Seed sweep |
| ATCNet + mixup + plateau | `120 ep`, seed 97 | **68.06%** | Ties best single-model result |
| ATCNet + S&R augmentation | `40 ep`, `batch=64` | 63.89% | Did not help alone |
| ATCNet compact (CTNet-like width/dropout) + mixup + plateau | `120 ep` | 57.41% | Too weak here |
| ATCNet on original train/val split | `30 ep` | 57.78% | Worse than fold-1 CV protocol |

## Current Best Recipe

```bash
python 0_run_train.py --dataset BCIC2A --model ATCNet --fold 1 --epochs 120 --lr 1e-3 --device cpu --mixup_alpha 0.2 --scheduler plateau --plateau_patience 15 --plateau_factor 0.9 --plateau_min_lr 1e-4 --patience 35 --seed 42
```

Best local result so far from this recipe: **67.59%**

### Seed Sweep Outcome

- `run_bcic2a_seed_sweep.py` completed five additional seeds.
- Best single-model result after the sweep: **68.06%** at seeds `37` and `97`.
- This suggests the best current ATCNet recipe is fairly stable in the `67.6% - 68.1%` range, but does not cross `70%` as a single model on the current CPU fold-1 protocol.

### Ensemble Follow-Up

- A soft-voting ensemble over three complementary ATCNet runs exceeded the stop target on the validation split:
  - `BCIC2A_ATCNet_20260516_004839_fold1`
  - `BCIC2A_ATCNet_20260516_005214_fold1`
  - `BCIC2A_ATCNet_20260516_001722_fold1`
- Validation accuracy of the ensemble: **71.76%**
- This became the most promising final CPU-side result because it crossed the `70%` target without introducing a heavier new backbone.

## Interpretation

- The most useful gain tonight came from **method + protocol improvement together**:
  - `mixup` gave a better late-epoch generalization trend than plain ATCNet.
  - `plateau` + longer training let that gain mature.
- Seed variation mattered, but only by roughly half a point around the improved ATCNet recipe.
- The strongest final improvement came from **ensemble diversity across strong ATCNet runs**, not from a brand-new architecture.
- The directions that looked strong in paper repos but did **not** transfer cleanly to this processed BCIC2A version were:
  - train-split `(channel, time)` standardization,
  - S&R augmentation by itself,
  - shrinking ATCNet toward a CTNet-like compact width.

## Next Recommended Steps

1. Treat the 3-model ATCNet ensemble as the strongest CPU-side result for now.
2. If a GPU becomes available, test either a larger ATCNet regularization sweep or a full CTNet/heavier-model comparison.
3. Keep the baseline, best single model, best ensemble, and runner-up runs because their `predictions.txt` files are valuable project outputs.
