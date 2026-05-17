# Results Summary — Organized by Dataset

## BCIC2A (Motor Imagery, 4-class, 22 channels)

### Best Results

| Run | Config | Best Val Acc | Notes |
|-----|--------|:---:|-------|
| seed 37 | ATCNet + mixup + plateau | **68.06%** | Best single model (fold 1, seed 37) |
| seed 97 | ATCNet + mixup + plateau | **68.06%** | Ties best (fold 1, seed 97) |
| 3-model ensemble | ATCNet seeds 37, 43, (none) | **71.76%** | Best ensemble result |

### Full Run Log

| Run Dir | Config | Best Val Acc | Seed |
|---------|--------|:---:|:----:|
| `BCIC2A_ATCNet_20260516_004839_fold1` | ATCNet + mixup + plateau, 120ep | **68.06%** | 37 |
| `BCIC2A_ATCNet_20260516_005606_fold1` | ATCNet + mixup + plateau, 120ep | **68.06%** | 97 |
| `BCIC2A_ATCNet_20260516_004005_fold1` | ATCNet + mixup + plateau, 120ep | 67.59% | 21 |
| `BCIC2A_ATCNet_20260516_004506_fold1` | ATCNet + mixup + plateau, 120ep | 67.59% | 29 |
| `BCIC2A_ATCNet_20260516_005214_fold1` | ATCNet + mixup + plateau, 120ep | 67.59% | 43 |
| `BCIC2A_ATCNet_20260516_001722_fold1` | ATCNet + mixup + plateau, 120ep | 67.59% | — |
| `BCIC2A_ATCNet_20260516_001239_fold1` | ATCNet + mixup + plateau, 120ep | 65.74% | — |
| `BCIC2A_ATCNet_20260516_000045_fold1` | ATCNet baseline, 35ep | 62.96% | — |
| `BCIC2A_ATCNet_20260516_000104_fold1` | ATCNet + mixup, 40ep | 65.74% | — |
| `BCIC2A_ATCNet_20260516_000318_fold1` | ATCNet + S&R aug, 40ep, batch=64 | 63.89% | — |
| `BCIC2A_ATCNet_20260516_000506_fold1` | ATCNet compact (CTNet-like), 120ep | 57.41% | — |
| `BCIC2A_ATCNet_20260516_001558` | ATCNet on original split, 30ep | 57.78% | — |
| `BCIC2A_ATCNet_20260516_002500_fold1` | ATCNet (F1=8, d_model=16, no mixup) | 57.41% | — |
| `BCIC2A_ATCNet_20260516_002945_fold1` | ATCNet + mixup + plateau | 67.59% | — |
| `BCIC2A_ATCNet_20260516_002053_fold1` | ATCNet + mixup + plateau, 120ep | 67.59% | — |
| `BCIC2A_ATCNet_ensemble_20260516_011256` | 3-model soft-voting ensemble | **71.76%** | — |

### New 2026-05-17 Changes (ATCNet++ & EEGConformer)

| Run Dir | Config | Val Acc (1 epoch) | Notes |
|---------|--------|:---:|-------|
| `BCIC2A_ATCNet_20260517_172542_fold1` | ATCNet-Large + grad_clip + warmup | 25.00% | Smoke test only |
| `BCIC2A_ATCNet_20260517_172707_fold1` | ATCNet-XL | 25.00% | Smoke test only |
| `BCIC2A_EEGConformer_20260517_172637_fold1` | EEGConformer (dim=64, 4 blocks) | 25.46% | Smoke test only |

See `design/BCIC2A-optimization-2026-05-16.md` for full details.

---

## SEED (Emotion Recognition, 3-class, 62 channels)

| Run Dir | Config | Best Val Acc | Notes |
|---------|--------|:---:|-------|
| `SEED_SUB1_DE_DGCNN_20260516_025234` | DGCNN on SEED_SUB1_DE | — | Single-subject DE |
| `SEED_SUB1_DE_RANDOM_DGCNN_20260516_023221` | DGCNN on SEED_SUB1_DE_RANDOM | — | Random baseline |
| `SEED_BYSUBJ_DGCNN_20260516_034834_allfolds` | DGCNN 5-fold CV, 60ep | 47.11% ± 2.11% | Cross-subject DGCNN |
| `SEED_codex_claude_20260516` | Various model smoke tests | — | See subdirectories |

---

## MDD (Depression Diagnosis, 2-class, 20 channels)

| Run Dir | Config | Best Val Acc | Notes |
|---------|--------|:---:|-------|
| `MDD_EEGLSTM_pretrained_20260514_230113` | SimCLR pretrain + EEGLSTM finetune | **87.50%** | Best MDD result |
