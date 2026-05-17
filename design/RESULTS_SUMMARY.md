# BCIC2A Optimization — Sweep Results Summary

## Overall Progress

| Stage | Best Single | Best Ensemble | Target |
|-------|:-----------:|:-------------:|:------:|
| Pre-sweep (CPU) | 68.06% | 71.76% | — |
| Round 1 | **69.91%** (weight_decay, seed 37) | — | — |
| Round 2 | **68.98%** (combined, seed 42) | 67.59% (3-model) | — |
| **Round 3** | *(running)* | *(running)* | **75%** |

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

## Round 3 — Standardization + 5-fold CV (RUNNING)

| Config | Changes | Seeds | Status |
|--------|---------|:-----:|:------:|
| 14-combined-large-lr5e4 | Large + wd + clip + warmup + lr=5e-4 | 21, 37 | Pending |
| 15-combined-std | wd + clip + warmup + `--standardize_inputs` | 21, 37 | Pending |
| 16-combined-std-noise | wd + clip + warmup + std + `--aug_noise_std 0.05` | 21, 37 | Pending |
| 17-eegconformer-v3 | Conformer + std + lr=1e-3 + plateau | 29, 43 | Pending |
| 18-5fold-combined | 5-fold CV with best recipe (seed 37) | 37 | Pending |

**Round 3 hypotheses:**
- Data isn't standardized → standardization `(--standardize_inputs)` fixes both aug_noise and Conformer
- 5-fold CV provides diverse models for effective ensemble
- Large preset with lower LR (5e-4) matches base model performance

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
