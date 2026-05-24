# MDD EEG Binary Classification - Run Report

## 1. Project Path

```
D:\ML\course project\course project\MDD
```

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.13.2 |
| PyTorch | 2.7.1+cpu |
| CUDA | Not available |
| h5py | 3.16.0 |
| numpy | 2.2.6 |
| sklearn | 1.7.0 |

## 3. Data Files Check

All data files present and valid. No NaN or Inf values detected.

| File | Keys | Status |
|------|------|--------|
| train.h5 | X, y | OK |
| val.h5 | X, y | OK |
| test_x_only.h5 | X | OK |
| dataset_info.json | - | OK |

## 4. Data Shapes

| Dataset | X Shape | y Shape | dtype |
|---------|---------|---------|-------|
| Train | (960, 20, 200) | (960,) | float32 / int64 |
| Val | (640, 20, 200) | (640,) | float32 / int64 |
| Test | (800, 20, 200) | N/A | float32 |

- 20 EEG channels (10-20 montage)
- 200 time points (1 second at 200 Hz sampling rate)
- Original sampling rate: 256 Hz, resampled to 200 Hz

## 5. Label Distribution

| Dataset | Class 0 (Healthy) | Class 1 (MDD) | Balance |
|---------|-------------------|---------------|---------|
| Train | 480 (50.0%) | 480 (50.0%) | Balanced |
| Val | 320 (50.0%) | 320 (50.0%) | Balanced |

## 6. Checkpoint Evaluation Results (All Models)

### Key Finding: Old checkpoints were trained on RAW (unnormalized) data

| Checkpoint | Model | Val Acc (raw) | Val Acc (normalized) | Best |
|---|---|---|---|---|
| hybrid_best_model.pth | EEGNetHybrid | **0.9391** | 0.5000 | 0.9391 |
| ensemble_model_seed_123.pth | EEGNet | **0.9344** | 0.5000 | 0.9344 |
| ensemble_model_seed_42.pth | EEGNet | **0.9297** | 0.5000 | 0.9297 |
| outputs/eegnet_seed42_best.pth | EEGNet | 0.5672 | **0.9297** | 0.9297 |
| ensemble_model_seed_2024.pth | EEGNet | **0.9266** | 0.5000 | 0.9266 |
| ultimate_best_model.pth | EEGNet | **0.9203** | 0.5000 | 0.9203 |
| best_model_fold_1.pth | EEGNet | **0.9156** | 0.5000 | 0.9156 |
| best_model_fold_2.pth | EEGNet | **0.9156** | 0.5000 | 0.9156 |
| best_model_fold_3.pth | EEGNet | **0.9047** | 0.5016 | 0.9047 |
| best_model.pth | EEGNetOld | **0.9031** | 0.5000 | 0.9031 |
| best_model_fold_4.pth | EEGNet | **0.8969** | 0.5000 | 0.8969 |
| best_model_fold_5.pth | EEGNet | **0.8828** | 0.5000 | 0.8828 |

## 7. Best Single Model

- **Checkpoint**: `hybrid_best_model.pth`
- **Architecture**: EEGNetHybrid (EEGNet + Differential Entropy frequency features)
- **Val Accuracy**: 0.9391
- **Val Balanced Accuracy**: 0.9391
- **Val Macro F1**: 0.9389
- **Input mode**: Raw (unnormalized)

## 8. Ensemble Evaluation (Top Strategies)

| Rank | Strategy | Val Acc | Val F1 |
|------|----------|---------|--------|
| 1 | **threshold-tuned top-3 weighted (t=0.61)** | **0.9563** | **0.9562** |
| 2 | top-2 average | 0.9484 | 0.9483 |
| 3 | acc>=0.93 (2 models) | 0.9484 | 0.9483 |
| 4 | best exhaustive combo | 0.9484 | 0.9483 |
| 5 | top-4 average | 0.9469 | 0.9468 |
| 6 | seed+hybrid (5 models) | 0.9422 | 0.9421 |
| 7 | single: hybrid_best_model.pth | 0.9391 | 0.9389 |

## 9. Final Adopted Strategy

**Threshold-tuned top-3 weighted ensemble (threshold=0.61)**

- Models used:
  1. `hybrid_best_model.pth` (EEGNetHybrid, weight=0.335, raw input)
  2. `ensemble_model_seed_123.pth` (EEGNet, weight=0.333, raw input)
  3. `ensemble_model_seed_42.pth` (EEGNet, weight=0.332, raw input)
- All inputs: raw (unnormalized) EEG data
- Decision: predict class 1 (MDD) if weighted prob >= 0.61
- **Val Accuracy: 0.9563**
- **Val Balanced Accuracy: 0.9563**
- **Val Macro F1: 0.9562**
- Confusion Matrix: [[301, 19], [9, 311]]

### Why ensemble over single model?

The threshold-tuned weighted ensemble (95.63%) outperforms the best single model (93.91%) by **+1.72%** absolute accuracy improvement. The ensemble combines diversity from:
- EEGNetHybrid (spatial + frequency features via DE)
- EEGNet with two different random seeds (architectural diversity through initialization)

## 10. Training Decision

**No retraining performed.**

Reason: Current machine has CPU only (no CUDA). The existing checkpoints achieve 95.63% val accuracy through ensemble, which is already strong. Retraining on CPU would be very slow and unlikely to improve significantly over current results.

## 11. TTA (Test Time Augmentation)

**Not used.**

Reason: Given the strong ensemble performance (95.63%) and CPU-only environment, TTA would add significant inference time without guaranteed improvement.

## 12. Threshold Tuning

**Yes, applied.**

Optimal threshold found via grid search on validation set: **t=0.61** (instead of default 0.5). This shifts the decision boundary slightly toward requiring higher confidence for MDD classification, reducing false positives.

## 13. Final MDD.txt Prediction Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 0 (Healthy Controls) | 326 | 40.8% |
| 1 (Major Depressive Disorder) | 474 | 59.2% |
| **Total** | **800** | 100% |

## 14. MDD.txt Format Check

- File exists: YES
- Lines: 800 (matches test samples)
- All values: 0 or 1
- No extra whitespace or headers
- **Format: PASS**

## 15. Recommendations for Further Improvement

1. **GPU Training**: Re-train with CUDA for faster iteration:
   ```bash
   python train.py --model eegnet --epochs 150 --batch-size 64 --lr 0.001 --seed 42 --augment
   python train.py --model eegnet --epochs 150 --batch-size 64 --lr 0.001 --seed 123 --augment
   python train.py --model eegnet --epochs 150 --batch-size 64 --lr 0.001 --seed 2024 --augment
   ```

2. **Larger Ensemble**: Train 5-10 seeds and select top-k for ensemble.

3. **Cross-Validation**: Implement proper 5-fold CV with subject-level splitting to avoid data leakage.

4. **Architecture Search**: Try deeper models (DeepConvNet, ShallowConvNet, EEGConformer).

5. **Frequency Features**: The hybrid model's advantage suggests spectral features are valuable. Consider adding wavelet features or more frequency bands.

6. **TTA**: Add test-time augmentation (time shift, amplitude scaling) for potential +0.5-1% gain.

7. **Stacking**: Train a meta-learner on top of base model predictions.

8. **Subject-level Evaluation**: Verify that high accuracy isn't due to within-subject correlation (different segments from same subject in both train and val).

---

*Report generated: 2026-05-23*
*Project: MDD EEG Binary Classification*
*Best Val Accuracy: 0.9563 (threshold-tuned weighted ensemble)*
