# BCIC2A Optimization — Sweep Results Summary

## Overall Progress

| Stage | Best Single | Best Ensemble | Target |
|-------|:-----------:|:-------------:|:------:|
| Pre-sweep (CPU) | 68.06% | 71.76% | — |
| Round 1 | **69.91%** (weight_decay, seed 37) | — | — |
| Round 2 | **68.98%** (combined, seed 42) | 67.59% (3-model) | — |
| Round 3 | **69.44%** (combined-std, seed 37) | 5-fold: **67.78%** mean | — |
| **Final** | **69.91%** (weight_decay s37) | **5-fold ensemble** *(pending)* | **75%** |

---

## Round 1 — Incremental improvements (fold 1, 300 epochs)

| Config | Changes | Seed 21 | Seed 37 | Seed 29 | Seed 43 |
|--------|---------|:-------:|:-------:|:-------:|:-------:|
| 01-baseline-original | Mixup + plateau (the 68% recipe) | 65.28% | 66.67% | — | — |
| 02-baseline-large | `--atc_preset large` | — | 64.35% | — | — |
| **03-weight-decay** | `--weight_decay 1e-4` | 64.35% | **69.91%** | — | — |
| 04-aug-ls | `--aug_noise_std 0.05` + `--label_smoothing 0.05` | 43.52% ❌ | 45.83% ❌ | — | — |
| **05-clip-warmup** | `--grad_clip_norm 1.0` + `--warmup_epochs 10` | 67.59% | **69.44%** | — | — |
| 06-full-combo | Large + wd + aug + ls + clip + warmup + bs64 | 42.13% ❌ | 46.30% ❌ | — | — |
| 07-eegconformer | Conformer, lr=5e-4, cosine, label_smoothing | — | — | 51.85% | 48.61% |

**Round 1 key findings:**
- `weight_decay=1e-4` (AdamW) with seed 37 achieves **69.91%** — new best single model
- `grad_clip_norm + warmup_epochs` consistently improves both seeds (67.59%, 69.44%)
- `aug_noise_std=0.05` + `label_smoothing=0.05` **collapses training to ~45%**
- Larger model alone (`atc_preset large`) without extra regularization underperforms (64.35%)
- EEGConformer at ~50% far below expectation

---

## Round 2 — Combine winners + isolate failures (fold 1, 300 ep)

| Config | Changes | Seed 21 | Seed 37 | Seed 42 |
|--------|---------|:-------:|:-------:|:-------:|
| 08-combined | wd + clip + warmup | 67.59% | 67.59%* | **68.98%** |
| 09-combined-large | wd + clip + warmup + large | 65.74% | 67.59% | 67.13% |
| 10-noise-only | `--aug_noise_std 0.05` alone | 43.06% ❌ | 47.22% ❌ | — |
| 11-ls-only | `--label_smoothing 0.05` alone | 64.35% | 67.13% | — |
| 12-eegconformer-v2 | Conformer, lr=2e-4, plateau, no ls | — | — | 50.00% (s29) / 53.70% (s43) |
| 13-baseline-seed42 | Baseline extra seed | — | — | 64.35% |

*\*seed 37 run hit a sweep bug (picked up ensemble dir instead of actual run dir)*

**Round 2 key findings:**
- **`aug_noise_std=0.05` is the sole root cause of all 45% failures** (config 10 = 43-47%)
- Label smoothing mildly degrades but doesn't collapse (64-67%)
- Combined recipe (wd + clip + warmup) peaks at 68.98% (seed 42)
- Large preset with regularization reaches 65-67% — better than Round 1's 64% but still below base model
- EEGConformer still at 50-53% — likely needs **data standardization**

---

## Round 3 — Standardization + 5-fold CV (fold 1, 300 ep unless noted)

| Config | Changes | Seed 21 | Seed 37 | Seed 42 | Seed 29 | Seed 43 |
|--------|---------|:-------:|:-------:|:-------:|:-------:|:-------:|
| 14-combined-large-lr5e4 | wd+clip+warmup+large+lr=5e-4 | 64.35% | 65.74% | — | — | — |
| **15-combined-std** | wd+clip+warmup+`--standardize_inputs` | **68.06%** | **69.44%** | — | — | — |
| 16-combined-std-noise | wd+clip+warmup+std+`--aug_noise_std 0.05` | 67.13% | 68.06% | — | — | — |
| 17-eegconformer-v3 | Conformer+std+lr=1e-3 | — | — | — | 36.11% ❌ | 37.04% ❌ |
| 18-5fold-combined | 5-fold CV with best recipe (s37) | — | **67.78%** (mean) | — | — | — |

**Round 3 key findings:**
- Standardization helps ATCNet (config 15: 68-69%), on par with best non-standardized runs
- `aug_noise_std` is benign with standardized data (config 16: 67-68%) — confirms hypothesis
- Large preset still underperforms (64-65%) — base model is BCIC2A sweet spot
- EEGConformer v3 (36-37%) worse than v2 — **Conformer path abandoned**
- 5-fold CV gives 5 diverse models at 67.78% mean — ready for ensemble

---

## Final Push: 5-fold Ensemble

Train 5 models on different fold splits → average test predictions for model diversity:

```bash
python run_bcic2a_ensemble_5fold.py Results/BCIC2A_ATCNet_20260517_044019_allfolds
```

This script loads each fold's `model.pt`, runs soft-voting on the test set, and saves the ensemble predictions. The 5 models were trained on different 80/20 splits, giving them diverse decision boundaries — the most promising path to 75%.

---

## Pre-sweep Results (for reference)

| Model | Config | Best Val Acc |
|-------|--------|:-----------:|
| ATCNet (seed 37) | mixup + plateau, 120 ep | 68.06% |
| ATCNet (seed 97) | mixup + plateau, 120 ep | 68.06% |
| ATCNet (seed 21) | mixup + plateau, 120 ep | 67.59% |
| ATCNet (seed 29) | mixup + plateau, 120 ep | 67.59% |
| ATCNet (seed 43) | mixup + plateau, 120 ep | 67.59% |
| 3-model ensemble | seeds 37, 43, 97 | **71.76%** |
